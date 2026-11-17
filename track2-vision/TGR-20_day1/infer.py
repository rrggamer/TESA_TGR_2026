
import argparse, csv, glob, os, shutil, time, cv2, numpy as np, torch
from PIL import Image
import joblib
from crop_beans import beans_from_image
try:                                              # รองรับทั้งชื่อ train.py และ train_cv.py
    from train import build_model, val_tf
except ImportError:
    from train_cv import build_model, val_tf

torch.set_num_threads(4)                          # Pi 4 มี 4 คอร์
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_model(weights_dir):
    ck = torch.load(os.path.join(weights_dir, "best.pt"), map_location="cpu")
    head = ck["head"]
    m = build_model(len(ck["classes"]), pretrained=False, head="mlp" if head in ("hybrid", "hand") else head)
    m.load_state_dict(ck["state_dict"]); m.eval()
    pipe = joblib.load(os.path.join(weights_dir, "head.joblib")) if head in ("hybrid", "hand") else None
    return (m, pipe, head), ck["classes"]


@torch.no_grad()
def predict_image(model, path, tta=True):
    m, pipe, head = model
    beans = beans_from_image(path)               # [(crop BGR 160x160, feature 12 ค่า), ...]
    if not beans:                                 # หาเมล็ดไม่เจอ → ใช้ทั้งภาพแทน
        img = cv2.imread(path); beans = [(cv2.resize(img, (160, 160)), [0.0] * 12)]
    crops = [c for c, _ in beans]; H = np.array([f for _, f in beans], np.float32)
    if head == "hand":                            # ไม่ใช้ CNN เลย
        P = torch.tensor(pipe.predict_proba(H)); return P.mean(0), len(beans), P, crops
    x = torch.stack([val_tf(Image.fromarray(c[:, :, ::-1])) for c in crops])   # BGR→RGB
    views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)] if tta else [x]
    if head == "hybrid":                          # feature CNN เฉลี่ยทุกมุม + feature สี/รูปทรง → sklearn
        X = torch.stack([m.avgpool(m.features(v)).flatten(1) for v in views]).mean(0).numpy()
        P = torch.tensor(pipe.predict_proba(np.hstack([X, H]))); return P.mean(0), len(beans), P, crops
    P = torch.stack([m(v).softmax(1) for v in views]).mean(0); return P.mean(0), len(beans), P, crops   # เฉลี่ยทุกเมล็ด = คำตอบของภาพ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--debug", action="store_true", help="แสดงผลทายรายเมล็ด + บันทึก crop ลง <out>/_debug/")
    a = ap.parse_args()
    model, classes = load_model(a.weights)
    for c in classes: os.makedirs(os.path.join(a.out, c), exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(a.input, "*")) if f.lower().endswith(EXTS))

    rows, times = [], []
    for f in files:
        t0 = time.perf_counter()
        prob, n, P, crops = predict_image(model, f, tta=not a.no_tta)
        times.append(time.perf_counter() - t0)
        pred = classes[int(prob.argmax())]
        name = os.path.basename(f)
        rows.append((name, pred))
        shutil.copy2(f, os.path.join(a.out, pred, name))
        print(f"{name}: {pred} ({prob.max():.2f}) | {n} beans | {times[-1] * 1000:.0f} ms")
        if a.debug:
            top = torch.topk(prob, min(3, len(classes)))
            print("   รวมทั้งภาพ top-3:", ", ".join(f"{classes[i]} {v:.2f}" for v, i in zip(top.values, top.indices)))
            dd = os.path.join(a.out, "_debug", os.path.splitext(name)[0]); os.makedirs(dd, exist_ok=True)
            for j, (pj, c) in enumerate(zip(P, crops)):
                cj = classes[int(pj.argmax())]
                print(f"   bean {j:02d}: {cj:<10} {pj.max():.2f}")
                cv2.imwrite(os.path.join(dd, f"{j:02d}_{cj}_{pj.max():.2f}.jpg"), c)

    with open(os.path.join(a.out, "prediction.csv"), "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["image_name", "prediction"]); w.writerows(rows)
    if times:
        print(f"\n{len(files)} images | mean {np.mean(times) * 1000:.0f} ms/image "
              f"| min {min(times) * 1000:.0f} | max {max(times) * 1000:.0f}")