"""Day 3 client: รับภาพทีละภาพทาง HTTP → infer (ในโปรเซสเดียว) → POST ผลทันที (Task A) + publish MQTT (Task B) ไม่รอกัน

เซิร์ฟเวอร์กรรมการ (TopGun Benchmark Server):
  GET  /api/v1/image?mode=train|test   → ภาพ 1 ภาพ + header X-Image-ID, X-Image-Name   (test มีโควตา!)
  POST /api/v1/submit  (header X-Image-ID, JSON {image_name, objects[id,class,polygon]})  → เซิร์ฟเวอร์จับเวลา GET→submit เอง
  GET  /api/v1/stats                    → สถิติทีม (ไม่กินโควตา)
  auth: Authorization: Bearer <token>

ตั้งค่าใน client_config.json แล้ว:
  python client.py --mode train                 # ซ้อม: ดึงภาพ train ทีละภาพจนหมด/ครบ max_images
  python client.py --mode test                  # ของจริง (โควตา 5 ครั้ง · ถามยืนยันก่อน)
  python client.py --stats                      # ดูสถิติทีม
  python client.py --mqtt-test                  # แค่ต่อ MQTT (ไม่ publish)
  python client.py --dry-run --input <folder>   # ทดสอบในเครื่อง ไม่ต่อเน็ตเลย
  python client.py --input <folder> --no-http   # infer ภาพในเครื่อง + publish MQTT
"""
import argparse, glob, json, os, queue, shutil, sys, threading, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_CFG = {
    "team": "XX",
    "http": {
        "base_url": "TODO",                         # เช่น https://edgeai-tesa.c333m.site
        "token": "TODO",
        "mode": "train",                            # train / test / null (= โหมดปัจจุบันของเซิร์ฟเวอร์)
        "max_images": {"train": 16, "test": 5},     # กันวนไม่จบ · test = โควตา
        "timeout": 60, "retries": 3,
    },
    "mqtt": {
        "url": "",                                  # เช่น "wss://host/mqtt/" (ทับ host/port/transport/ws_path/tls)
        "host": "TODO", "port": 1883, "transport": "tcp", "ws_path": "/mqtt",
        "username": "", "password": "",
        "topic": "TODO",                            # ใส่ {team} {image} {track} {event} ได้
        "topic_vars": {},
        "client_id": "", "qos": 1, "tls": False, "keepalive": 60,
    },
}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path):
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    if os.path.isfile(path):
        for k, v in json.load(open(path, encoding="utf-8")).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
            else: cfg[k] = v
    return cfg


def need(value, name):
    if not value or "TODO" in str(value): sys.exit(f"ยังไม่ได้ตั้ง {name} ใน client_config.json")
    return value


# ───────────────────────── HTTP ─────────────────────────
class Server:
    def __init__(self, cfg):
        import requests
        h = cfg["http"]; self.h = h
        self.base = need(h["base_url"], "http.base_url").rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = "Bearer " + need(h["token"], "http.token")

    def stats(self):
        r = self.s.get(self.base + "/api/v1/stats", timeout=self.h["timeout"]); r.raise_for_status(); return r.json()

    def get_image(self, mode, retries):
        """→ (image_id, name, bytes) หรือ None ถ้าเซิร์ฟเวอร์ไม่ส่งภาพ (หมด / โควตาเต็ม / paused)"""
        params = {"mode": mode} if mode else {}
        for k in range(1, retries + 1):
            try:
                r = self.s.get(self.base + "/api/v1/image", params=params, timeout=self.h["timeout"])
                break
            except Exception as e:
                log(f"GET image ล้มเหลว ({k}/{retries}): {e}")
                if k == retries: raise
                time.sleep(1)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or not ctype.startswith("image/"):
            log(f"เซิร์ฟเวอร์ไม่ส่งภาพ: HTTP {r.status_code} {ctype} {r.text[:300]}"); return None
        iid = r.headers.get("X-Image-ID"); name = r.headers.get("X-Image-Name") or f"{iid}.jpg"
        return iid, os.path.basename(name), r.content

    def submit(self, iid, rec):
        for k in range(1, self.h["retries"] + 1):
            try:
                r = self.s.post(self.base + "/api/v1/submit", headers={"X-Image-ID": iid}, json=rec, timeout=self.h["timeout"])
                if r.status_code < 500: return r.status_code, r.text[:300]
                log(f"submit {iid}: HTTP {r.status_code} ({k}) {r.text[:200]}")
            except Exception as e:
                log(f"submit {iid} ล้มเหลว ({k}): {e}")
            time.sleep(0.5)
        return None, "submit ไม่สำเร็จ"


# ───────────────────────── MQTT ─────────────────────────
def mqtt_params(m):
    if m.get("url"):
        from urllib.parse import urlparse
        u = urlparse(m["url"]); sch = u.scheme.lower()
        ws = sch in ("ws", "wss"); tls = sch in ("wss", "mqtts", "ssl")
        port = u.port or (443 if sch == "wss" else 80 if sch == "ws" else 8883 if tls else 1883)
        return u.hostname, port, "websockets" if ws else "tcp", u.path or "/mqtt", tls
    return need(m["host"], "mqtt.host"), int(m["port"]), m.get("transport", "tcp"), m.get("ws_path", "/mqtt"), bool(m["tls"])


def mqtt_connect(cfg):
    """เชื่อม broker แล้วรอ CONNACK จริง (รหัสผ่านผิดจะรู้ทันที)"""
    import paho.mqtt.client as mqtt
    m = cfg["mqtt"]; host, port, transport, path, tls = mqtt_params(m)
    cid = m["client_id"] or f"team_{cfg['team']}_{int(time.time())}"
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid, transport=transport)
    if transport == "websockets": c.ws_set_options(path=path)
    if m["username"]: c.username_pw_set(m["username"], m["password"])
    if tls: c.tls_set()
    ev, res = threading.Event(), {}
    def on_connect(cl, ud, flags, rc, props=None): res["rc"] = rc; ev.set()
    c.on_connect = on_connect
    log(f"MQTT connect {transport}{'+tls' if tls else ''} {host}:{port}{path if transport == 'websockets' else ''} client_id={cid}")
    c.connect(host, port, int(m["keepalive"])); c.loop_start()
    if not ev.wait(20): c.loop_stop(); raise TimeoutError("ไม่ได้ CONNACK ภายใน 20 s")
    if res["rc"].is_failure: c.loop_stop(); c.disconnect(); raise ConnectionError(f"broker ปฏิเสธ: {res['rc']}")
    log(f"MQTT connected ({res['rc']})")
    return c


def mqtt_topic(cfg, image):
    m = cfg["mqtt"]; v = {"team": cfg["team"], "image": os.path.splitext(image)[0], **m.get("topic_vars", {})}
    t = m["topic"].format_map(v)
    if "TODO" in t: sys.exit(f"topic ยังมี TODO: {t} → ตั้ง mqtt.topic_vars ใน client_config.json")
    return t


class Publisher(threading.Thread):
    """publish ใน thread แยก → ไม่หน่วงการ submit HTTP"""
    def __init__(self, cfg):
        super().__init__(daemon=True); self.cfg = cfg; self.q = queue.Queue(); self.ok = 0; self.n = 0
        mqtt_topic(cfg, "x.jpg"); self.c = mqtt_connect(cfg)

    def run(self):
        while True:
            msg = self.q.get()
            if msg is None: break
            self.n += 1
            try:
                info = self.c.publish(mqtt_topic(self.cfg, msg["image"]), json.dumps(msg, ensure_ascii=False), qos=int(self.cfg["mqtt"]["qos"]))
                info.wait_for_publish(timeout=15)
                if info.is_published(): self.ok += 1; log(f"  MQTT ✓ {msg['image']}")
                else: log(f"  MQTT ✗ {msg['image']} (rc={info.rc})")
            except Exception as e:
                log(f"  MQTT ✗ {msg['image']}: {e}")

    def close(self):
        self.q.put(None); self.join(60); self.c.loop_stop(); self.c.disconnect()


# ───────────────────────── infer ─────────────────────────
class Infer:
    def __init__(self, weights, out):
        import infer, numpy as np, cv2
        self.m_ = infer; self.out = out
        cfg = infer.load_cfg(weights, argparse.Namespace())
        self.cfg = cfg; cv2.setNumThreads(cfg["threads"])
        for d in ("seg_mask", "vis", "mqtt"): os.makedirs(os.path.join(out, d), exist_ok=True)
        self.model = infer.load_model(weights, cfg)
        self.model.forward(np.full((self.model.S, self.model.S, 3), 114, np.uint8))     # warm-up
        log(f"model พร้อม: {os.path.join(weights, cfg['ncnn'])} imgsz={self.model.S}")

    def __call__(self, path):
        t0 = time.perf_counter(); rec, _ = self.m_.process(self.model, path, self.cfg, self.out); dt = time.perf_counter() - t0
        msg = self.m_.mqtt_message(rec, dt)
        with open(os.path.join(self.out, "mqtt", os.path.splitext(rec["image_name"])[0] + ".json"), "w", encoding="utf-8") as f:
            json.dump(msg, f, ensure_ascii=False, indent=2)
        return rec, msg, dt


def write_results(out, preds):
    s = json.dumps(preds, ensure_ascii=False, indent=2)
    for n in ("predictions.json", "results.json"):
        with open(os.path.join(out, n), "w", encoding="utf-8") as f: f.write(s)


def build_zip(cfg, out, weights):
    root = f"team_{cfg['team']}_day3"; zpath = os.path.join(HERE, f"team_{cfg['team']}_Day3.zip")
    files = [f for f in ("README.md", "requirements.txt") if os.path.isfile(os.path.join(HERE, f))]
    files += [os.path.basename(f) for f in glob.glob(os.path.join(HERE, "*.py"))]
    wd = os.path.abspath(weights); wj = os.path.join(wd, "infer_config.json")
    nd = os.path.join(wd, (json.load(open(wj, encoding="utf-8")) if os.path.isfile(wj) else {}).get("ncnn", "best_ncnn_model"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(os.path.join(HERE, f), f"{root}/{f}")
        for f in ("best.pt", "infer_config.json"):
            if os.path.isfile(os.path.join(wd, f)): z.write(os.path.join(wd, f), f"{root}/weights/{f}")
        for f in glob.glob(os.path.join(nd, "*")):
            if os.path.isfile(f): z.write(f, f"{root}/weights/{os.path.basename(nd)}/{os.path.basename(f)}")
        for dp, _, fs in os.walk(out):
            for f in fs:
                if f == "results.json": continue
                p = os.path.join(dp, f); z.write(p, f"{root}/outputs/" + os.path.relpath(p, out).replace(os.sep, "/"))
    log(f"zip: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TGR-20 Day 3 client (HTTP + MQTT)")
    ap.add_argument("--config", default=os.path.join(HERE, "client_config.json"))
    ap.add_argument("--mode", choices=["train", "test", "server"], help="ทับ http.mode (server = ตามโหมดเซิร์ฟเวอร์)")
    ap.add_argument("--max-images", type=int, help="จำนวนภาพสูงสุดที่จะดึง")
    ap.add_argument("--input", help="ใช้ภาพในโฟลเดอร์นี้แทนการดึงจากเซิร์ฟเวอร์ (ไม่ submit HTTP)")
    ap.add_argument("--test-dir", default=os.path.join(HERE, "test_data"), help="ที่เก็บภาพที่ดึงมา")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--no-http", action="store_true"); ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--no-zip", action="store_true", help="ไม่สร้าง team_XX_Day3.zip ตอนจบ")
    ap.add_argument("--dry-run", action="store_true", help="ไม่ต่อเครือข่ายเลย (ต้องใช้ --input)")
    ap.add_argument("--yes", action="store_true", help="ไม่ถามยืนยันก่อนใช้โหมด test")
    ap.add_argument("--stats", action="store_true", help="แสดงสถิติทีมแล้วจบ")
    ap.add_argument("--mqtt-test", action="store_true", help="แค่ต่อ MQTT แล้วตัด (ไม่ publish)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    if a.stats:
        print(json.dumps(Server(cfg).stats(), ensure_ascii=False, indent=2)); return 0
    if a.mqtt_test:
        c = mqtt_connect(cfg); c.loop_stop(); c.disconnect()
        log(f"MQTT test ผ่าน · topic: {mqtt_topic(cfg, '<image>.jpg')}"); return 0
    if a.dry_run:
        if not a.input: sys.exit("--dry-run ต้องใช้ --input <โฟลเดอร์ภาพ>")
        a.no_http = a.no_mqtt = True
    local = bool(a.input)
    mode = a.mode or cfg["http"].get("mode") or "server"
    mode = None if mode == "server" else mode
    if not local and a.no_http: sys.exit("--no-http ต้องใช้คู่กับ --input")
    if not local and mode != "train" and not a.yes:
        if input(f"โหมด {mode or 'ตามเซิร์ฟเวอร์'} อาจกินโควตา test (5 ครั้ง) · พิมพ์ yes เพื่อเริ่ม: ").strip().lower() != "yes":
            sys.exit("ยกเลิก")
    maxn = a.max_images or (cfg["http"]["max_images"].get(mode or "test", 5) if not local else 10 ** 9)

    if os.path.isdir(a.out): shutil.rmtree(a.out)                              # กันผลเก่าปน
    eng = Infer(a.weights, a.out)                                              # โหลดโมเดล + ต่อ MQTT ก่อนเริ่มนับเวลา
    srv = None if local else Server(cfg)
    pub = None if a.no_mqtt else Publisher(cfg)
    if pub: pub.start()

    preds, times, subm, t_all = [], [], [], time.time()
    if local:
        files = sorted(os.path.join(a.input, f) for f in os.listdir(a.input) if f.lower().endswith(IMG_EXT))
        if not files: sys.exit(f"ไม่พบภาพใน {a.input}")
        src = ((None, os.path.basename(f), f) for f in files)
    else:
        if os.path.isdir(a.test_dir): shutil.rmtree(a.test_dir)
        os.makedirs(a.test_dir)
        def gen():
            seen = set()
            for _ in range(maxn):
                g = srv.get_image(mode, 1 if mode != "train" else cfg["http"]["retries"])
                if g is None: return
                iid, name, data = g                                            # name = ชื่อจริงจากเซิร์ฟเวอร์ (ใช้ตอน submit/MQTT)
                local = name if name not in seen else f"{os.path.splitext(name)[0]}_{iid}{os.path.splitext(name)[1]}"
                seen.add(name); p = os.path.join(a.test_dir, local)            # ชื่อไฟล์ในเครื่องกันซ้ำ (train ส่งภาพซ้ำได้)
                with open(p, "wb") as f: f.write(data)
                yield iid, name, p
        src = gen()

    for iid, name, path in src:
        try:
            rec, msg, dt = eng(path)
        except Exception as e:
            log(f"{name}: infer ล้มเหลว {e}"); rec = {"image_name": name, "objects": []}; msg = None; dt = 0
        rec["image_name"] = name
        if msg: msg["image"] = name
        line =f"{name}: {len(rec['objects'])} เม็ด infer {dt * 1000:.0f} ms"
        if srv:
            code, txt = srv.submit(iid, rec); subm.append(code == 200)
            line += f" · submit {iid} → {code} {txt[:120]}"
        log(line)
        if pub and msg: pub.q.put(msg)
        preds.append(rec); times.append(dt)

    if pub: pub.close()
    if not preds: log("ไม่ได้ประมวลผลภาพเลย"); return 1
    write_results(a.out, preds)
    if not a.no_zip: build_zip(cfg, a.out, a.weights)
    import numpy as np
    log(f"จบ {len(preds)} ภาพ ใน {time.time() - t_all:.1f} s · infer median {np.median(times) * 1000:.0f} ms"
        + (f" · submit สำเร็จ {sum(subm)}/{len(subm)}" if srv else "")
        + (f" · MQTT {pub.ok}/{pub.n}" if pub else ""))
    if srv:
        try: log("stats: " + json.dumps(srv.stats(), ensure_ascii=False))
        except Exception as e: log(f"อ่าน stats ไม่ได้: {e}")
    return 0 if (not srv or all(subm)) and (not pub or pub.ok == pub.n) else 1


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    sys.exit(main())
