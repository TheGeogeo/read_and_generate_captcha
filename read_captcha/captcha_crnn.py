#!/usr/bin/env python3
from __future__ import annotations
import argparse, string, random
from pathlib import Path
from typing import List
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Constants -----------------------------
ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
BLANK = "_"
CHARS = BLANK + ALPHABET
C2I = {c: i for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}
IMG_H = 32
BATCH = 32
LR = 1e-3

# --------------------------- Pre‑processing ---------------------------

def maybe_invert(img: np.ndarray) -> np.ndarray:
    return 255 - img if img.mean() < 100 else img


def augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if random.random() < 0.3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), 1)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if random.random() < 0.3:
        tx, ty = random.randint(-2, 2), random.randint(-2, 2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def preprocess(path: Path, train: bool = False) -> torch.Tensor:
    g = cv2.imread(str(path), 0)
    if g is None:
        raise FileNotFoundError(path)
    g = maybe_invert(g)
    if train:
        g = augment(g)
    h, w = g.shape
    g = cv2.resize(g, (int(w * IMG_H / h), IMG_H), cv2.INTER_AREA)
    g = cv2.equalizeHist(g)
    return torch.from_numpy(g.astype("float32") / 255.0).unsqueeze(0)

# ------------------------------- Dataset -----------------------------

def txt2idx(txt: str) -> List[int]:
    return [C2I[c] for c in txt]


def idx2txt(seq: List[int]) -> str:
    return "".join(I2C[i] for i in seq if i)


class CaptchaDS(Dataset):
    def __init__(self, root: Path, train: bool = False):
        self.files = sorted([p for ext in ("*.png", "*.jpg", "*.jpeg") for p in root.glob(ext)])
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        return preprocess(p, self.train), torch.tensor(txt2idx(p.stem)), p.stem


def collate(batch):
    imgs, tgts, labels = zip(*batch)
    max_w = max(i.shape[-1] for i in imgs)
    imgs = torch.stack([F.pad(i, (0, max_w - i.shape[-1]), value=1.0) for i in imgs])
    tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    return imgs, torch.cat(tgts), tgt_lens, labels

# -------------------------------- Model ------------------------------

class CRNN(nn.Module):
    def __init__(self, n_class: int = len(CHARS)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):  # x: [B, 1, H, W]
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)  # [W', B, 512]
        f, _ = self.rnn(f)
        return self.fc(f)

# ------------------------------- Decode ------------------------------

def decode(logits: torch.Tensor) -> str:
    _, idx = logits.softmax(2).max(2)
    seq, prev = [], 0
    for i in idx.squeeze(1).tolist():
        if i != prev and i != 0:
            seq.append(i)
        prev = i
    return idx2txt(seq)

# ------------------------------ Evaluate -----------------------------

def evaluate(net: nn.Module, loader: DataLoader, crit, dev):
    net.eval(); total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, tgts, tgt_lens, labels in loader:
            imgs, tgts, tgt_lens = imgs.to(dev), tgts.to(dev), tgt_lens.to(dev)
            logits = net(imgs).log_softmax(2)
            T, B, _ = logits.size(); inp_lens = torch.full((B,), T, dtype=torch.long, device=dev)
            total += crit(logits, tgts, inp_lens, tgt_lens).item()
            for b in range(B):
                if decode(logits[:, b:b + 1, :]) == labels[b]:
                    correct += 1
                n += 1
    return total / len(loader), correct / n if n else 0.0

# -------------------------------- Train ------------------------------

def train(data: Path, epochs: int, out: Path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Using device:", dev)
    train_ld = DataLoader(CaptchaDS(data, True), BATCH, True, collate_fn=collate, num_workers=2)
    val_dir = data / "val"
    val_ld = DataLoader(CaptchaDS(val_dir), BATCH, False, collate_fn=collate, num_workers=2) if val_dir.exists() else None

    net, crit = CRNN().to(dev), nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(net.parameters(), LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)

    for ep in range(1, epochs + 1):
        net.train(); running = 0.0
        for imgs, tgts, tgt_lens, _ in train_ld:
            imgs, tgts, tgt_lens = imgs.to(dev), tgts.to(dev), tgt_lens.to(dev)
            logits = net(imgs).log_softmax(2)
            T, B, _ = logits.size(); inp_lens = torch.full((B,), T, dtype=torch.long, device=dev)
            loss = crit(logits, tgts, inp_lens, tgt_lens)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        train_loss = running / len(train_ld)

        if val_ld:
            val_loss, val_acc = evaluate(net, val_ld, crit, dev)
            print(f"Ep {ep}/{epochs} train={train_loss:.3f} val={val_loss:.3f} acc={val_acc:.2%}")
            sched.step(val_loss)
        else:
            print(f"Ep {ep}/{epochs} train={train_loss:.3f}")
            sched.step(train_loss)

    torch.save(net.state_dict(), str(out))

# ------------------------------- Predict -----------------------------

def predict(model: Path, pattern: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", dev)
    net = CRNN().to(dev)
    state = torch.load(model, map_location=dev, weights_only=True)
    net.load_state_dict(state)
    net.eval()

    for p in sorted(Path().glob(pattern)):
        img = preprocess(p).unsqueeze(0).to(dev)
        with torch.no_grad():
            txt = decode(net(img))
        print(f"{p.name:<20s} -> {txt}")

# --------------------------------- CLI --------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--model", type=Path, default=Path("model.pth"))

    pr = sub.add_parser("predict")
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--imgs", type=str, default="captchas/*.png")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.data, args.epochs, args.model)
    else:
        predict(args.model, args.imgs)
