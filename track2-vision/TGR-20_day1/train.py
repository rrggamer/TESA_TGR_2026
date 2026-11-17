
import argparse, glob, os, random, numpy as np, torch, torch.nn as nn, torchvision as tv
from PIL import Image
from torchvision import transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import csv, joblib

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 160
PCA_N = 0                                        # >0: ย่อ feature CNN เหลือ N มิติก่อนรวมกับ feature สี/รูปทรง (ตั้งผ่าน --pca)
FT = {"e1": 5, "e2": 25, "lr_bb": 1e-4}                # epoch ขั้น 1 / ขั้น 2 / lr ของ backbone (ตั้งผ่าน --e1 --e2 --lr-bb)
C_PROBE = 0.5                                    # ค่า C ของ Logistic Regression (ตั้งผ่าน --C)
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
train_tf = T.Compose([
    T.RandomRotation(180, fill=200), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.ColorJitter(0.2, 0.2, 0.1, 0.02),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    T.ToTensor(), T.Normalize(MEAN, STD)])
val_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def list_crops(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    items = []                                   # (path, label, group=ภาพต้นฉบับ, fold)
    for ci, c in enumerate(classes):
        files = sorted(glob.glob(os.path.join(root, c, "*.jpg")))
        groups = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in files})
        for f in files:
            g = os.path.basename(f).rsplit("_", 1)[0]
            items.append((f, ci, f"{c}/{g}", groups.index(g)))
    return classes, items


def build_model(n_cls, pretrained=True, head="mlp"):
    m = tv.models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
    if head == "linear":                         # สำหรับ probe: avgpool 576 -> n_cls
        m.classifier = nn.Linear(576, n_cls)
    else:
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
    return m


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y, *_ = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


@torch.no_grad()
def tta_probs(m, items, n_cls):                  # TTA: หมุน 4 ทิศ + flip แล้วเฉลี่ย
    m = m.to(DEV).eval(); out = []
    for p, *_ in items:
        x = val_tf(Image.open(p).convert("RGB"))
        xs = torch.stack([torch.rot90(v, k, (1, 2)) for v in (x, x.flip(2)) for k in range(4)])
        out.append(m(xs.to(DEV)).softmax(1).mean(0).cpu())
    return torch.stack(out).numpy()


@torch.no_grad()
def tta_feats(backbone, xs_list):              # xs_list: list ของ tensor [3,H,W] → feature เฉลี่ย 8 มุม ต่อเมล็ด
    backbone = backbone.to(DEV).eval(); out = []
    for x in xs_list:
        v = torch.stack([torch.rot90(t, k, (1, 2)) for t in (x, x.flip(2)) for k in range(4)]).to(DEV)
        out.append(backbone.avgpool(backbone.features(v)).flatten(1).mean(0).cpu().numpy())
    return np.array(out)


HAND = {}                                        # "class/file.jpg" -> feature สี+รูปทรง 12 ค่า (จาก crop_beans.py)
def load_hand(root):
    with open(os.path.join(root, "features.csv")) as fp:
        r = csv.reader(fp); next(r)
        return {row[0]: np.array(row[1:], np.float32) for row in r}

def hand_of(items):
    return np.stack([HAND[f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"] for p, *_ in items])


class Hybrid:                                    # CNN feature (+ feature สี/รูปทรง) → StandardScaler → LogisticRegression
    def __init__(self, backbone, pipe, use_cnn=True): self.bb, self.pipe, self.use_cnn = backbone, pipe, use_cnn
    def predict_items(self, items):
        H = hand_of(items)
        if not self.use_cnn: return self.pipe.predict_proba(H)
        X = tta_feats(self.bb, [val_tf(Image.open(p).convert("RGB")) for p, *_ in items])
        return self.pipe.predict_proba(np.hstack([X, H]))


def hybrid(tr, n_cls, pretrained, reps=10):
    bb = build_model(n_cls, pretrained)
    X, Y = features(bb, tr, train_tf, reps=reps)          # วน reps รอบ × ทุก item → ลำดับตรงกับ np.tile
    H = np.tile(hand_of(tr), (reps, 1))
    d = X.shape[1]
    if PCA_N > 0:                                          # ไม่ให้ CNN 576 มิติกลบ feature สี/รูปทรง 12 มิติ
        pre = ColumnTransformer([("cnn", make_pipeline(StandardScaler(), PCA(PCA_N, random_state=0)), slice(0, d)),
                                 ("hand", StandardScaler(), slice(d, d + H.shape[1]))])
    else:
        pre = StandardScaler()
    pipe = make_pipeline(pre, LogisticRegression(C=C_PROBE, max_iter=5000)).fit(np.hstack([X, H]), Y)
    return Hybrid(bb.cpu(), pipe)


def hand_only(tr, n_cls, pretrained):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C_PROBE, max_iter=5000))
    pipe.fit(hand_of(tr), np.array([it[1] for it in tr]))
    return Hybrid(None, pipe, use_cnn=False)


def report(probs, items, tag):
    y = np.array([it[1] for it in items]); groups = np.array([it[2] for it in items])
    bean = (probs.argmax(1) == y).mean()
    img_ok = [probs[groups == g].mean(0).argmax() == y[groups == g][0] for g in np.unique(groups)]
    print(f"{tag}: bean acc {bean:.3f} | image acc {np.mean(img_ok):.3f} ({sum(img_ok)}/{len(img_ok)})")
    return bean, np.mean(img_ok), y, probs.argmax(1)


def print_confusion(y, p, classes):              # แถว = คลาสจริง, คอลัมน์ = ที่ทาย
    n = len(classes); cm = np.zeros((n, n), int)
    for t, q in zip(y, p): cm[t, q] += 1
    w = max(len(c) for c in classes)
    print("\nconfusion (bean level, รวม 3 fold)  แถว=จริง คอลัมน์=ทาย")
    print(" " * (w + 1) + " ".join(c[:5].rjust(5) for c in classes) + "   recall")
    for i, c in enumerate(classes):
        print(c.ljust(w), " ".join(str(v).rjust(5) for v in cm[i]), f"  {cm[i, i] / max(cm[i].sum(), 1):.2f}")


def finetune(tr, n_cls, pretrained):
    e1, e2 = FT["e1"], FT["e2"]
    m = build_model(n_cls, pretrained).to(DEV)
    dl = torch.utils.data.DataLoader(DS(tr, train_tf), batch_size=16, shuffle=True, num_workers=2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    m.features.requires_grad_(False)             # ขั้น 1: ตรึง backbone
    opt = torch.optim.AdamW(m.classifier.parameters(), lr=1e-3)
    for ep in range(e1 + e2):
        if ep == e1:                             # ขั้น 2: ปลด backbone, lr ต่ำ
            m.features.requires_grad_(True)
            opt = torch.optim.AdamW([{"params": m.features.parameters(), "lr": FT["lr_bb"]},
                                     {"params": m.classifier.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        m.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            loss = crit(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    return m.cpu()


@torch.no_grad()
def features(backbone, items, tf, reps=1):
    backbone = backbone.to(DEV).eval(); X, Y = [], []
    for _ in range(reps):
        for p, y, *_ in items:
            x = tf(Image.open(p).convert("RGB"))[None].to(DEV)
            X.append(backbone.avgpool(backbone.features(x)).flatten(1)[0].cpu().numpy()); Y.append(y)
    return np.array(X), np.array(Y)


def probe(tr, n_cls, pretrained):
    bb = build_model(n_cls, pretrained)
    X, Y = features(bb, tr, train_tf, reps=10)   # augment 10 รอบต่อเมล็ด
    bb = bb.cpu()
    lr = LogisticRegression(C=C_PROBE, max_iter=3000).fit(X, Y)
    m = build_model(n_cls, pretrained, head="linear"); m.load_state_dict(bb.state_dict(), strict=False)
    m.classifier.weight.data = torch.tensor(lr.coef_, dtype=torch.float32)
    m.classifier.bias.data = torch.tensor(lr.intercept_, dtype=torch.float32)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops"); ap.add_argument("--mode", choices=["probe", "finetune", "hybrid", "hand"], default="probe")
    ap.add_argument("--final", action="store_true"); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--C", type=float, default=0.5); ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--e1", type=int, default=5); ap.add_argument("--e2", type=int, default=25)
    ap.add_argument("--lr-bb", type=float, default=1e-4)
    ap.add_argument("--wdir", default="weights", help="โฟลเดอร์ที่บันทึก weight (กันเขียนทับของเดิม)")
    a = ap.parse_args(); C_PROBE = a.C; PCA_N = a.pca; FT.update(e1=a.e1, e2=a.e2, lr_bb=a.lr_bb)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    classes, items = list_crops(a.crops); n = len(classes)
    print("device:", DEV, "| classes:", classes, "| crops:", len(items))
    if a.mode in ("hybrid", "hand"): HAND.update(load_hand(a.crops))
    train_fn = {"probe": probe, "finetune": finetune, "hybrid": hybrid, "hand": hand_only}[a.mode]
    if a.final:
        m = train_fn(items, n, not a.no_pretrained)
        os.makedirs(a.wdir, exist_ok=True)
        if a.mode in ("hybrid", "hand"):                   # backbone → best.pt, หัว sklearn → head.joblib
            joblib.dump(m.pipe, os.path.join(a.wdir, "head.joblib"))
            bb = m.bb if m.bb is not None else build_model(n, False)
            torch.save({"state_dict": bb.state_dict(), "classes": classes, "head": a.mode, "size": SIZE}, os.path.join(a.wdir, "best.pt"))
            print(f"saved {a.wdir}/best.pt + {a.wdir}/head.joblib")
        else:
            torch.save({"state_dict": m.state_dict(), "classes": classes,
                        "head": "linear" if a.mode == "probe" else "mlp", "size": SIZE}, os.path.join(a.wdir, "best.pt"))
            print(f"saved {a.wdir}/best.pt")
    else:
        res = []
        for k in range(3):
            tr = [it for it in items if it[3] != k]; va = [it for it in items if it[3] == k]
            m = train_fn(tr, n, not a.no_pretrained)
            probs = m.predict_items(va) if isinstance(m, Hybrid) else tta_probs(m, va, n)
            res.append(report(probs, va, f"fold {k}"))
        print(f"MEAN: bean acc {np.mean([r[0] for r in res]):.3f} | image acc {np.mean([r[1] for r in res]):.3f}")
        print_confusion(np.concatenate([r[2] for r in res]), np.concatenate([r[3] for r in res]), classes)