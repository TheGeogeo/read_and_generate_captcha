#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, string
from pathlib import Path
from typing import List
import cv2, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
BLANK = "_"
CHARS = BLANK + ALPHABET
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
IMG_H = 32
BATCH = 32
LR = 1e-3

# ---------- preprocessing ----------

def preprocess(p: Path) -> torch.Tensor:
    g = cv2.imread(str(p), 0)
    if g is None:
        raise FileNotFoundError(p)
    # ensure letters are dark (low values), background light; invert only if background darker
    if g.mean() < 127:
        g = 255 - g
    h, w = g.shape
    g = cv2.resize(g, (int(w * IMG_H / h), IMG_H), cv2.INTER_AREA)
    return torch.from_numpy(g.astype("float32") / 255.0).unsqueeze(0)

# ---------- dataset ---------------

def t2i(t: str) -> List[int]:
    return [CHAR2IDX[c] for c in t]

def i2t(v: List[int]) -> str:
    return "".join(IDX2CHAR[i] for i in v if i)

class CaptchaDS(Dataset):
    def __init__(self, root: Path):
        self.files = sorted([q for ext in ("*.png", "*.jpg", "*.jpeg") for q in root.glob(ext)])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]
        return preprocess(p), torch.tensor(t2i(p.stem)), p.stem

def pad(batch):
    imgs, tgts, labels = zip(*batch)
    mw = max(i.shape[-1] for i in imgs)
    imgs = torch.stack([F.pad(i, (0, mw - i.shape[-1]), value=1.0) for i in imgs])
    return imgs, torch.cat(tgts), torch.tensor([len(t) for t in tgts]), labels

# ---------- model -----------------

class CRNN(nn.Module):
    def __init__(self, n: int = len(CHARS)):
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
        self.fc = nn.Linear(512, n)
    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2, 0, 1)
        f, _ = self.rnn(f)
        return self.fc(f)

# ---------- utils -----------------

def decode(logits: torch.Tensor) -> str:
    _, idx = logits.softmax(2).max(2)
    seq, prev = [], 0
    for i in idx.squeeze(1).tolist():
        if i != prev and i != 0:
            seq.append(i)
        prev = i
    return i2t(seq)

def eval_loop(net, loader, crit, dev):
    net.eval(); tot = 0.0; ok = 0; tot_n = 0
    with torch.no_grad():
        for imgs, tgts, lens, labels in loader:
            imgs, tgts, lens = imgs.to(dev), tgts.to(dev), lens.to(dev)
            out = net(imgs).log_softmax(2)
            T, B, _ = out.size(); inp = torch.full((B,), T, dtype=torch.long, device=dev)
            tot += crit(out, tgts, inp, lens).item()
            for b in range(B):
                if decode(out[:, b:b+1, :]) == labels[b]:
                    ok += 1
                tot_n += 1
    return tot / len(loader), ok / tot_n if tot_n else 0.0

# ---------- train -----------------

def train(data: Path, epochs: int, sav: Path):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:', dev)
    tr_ld = DataLoader(CaptchaDS(data), BATCH, True, collate_fn=pad, num_workers=2)
    val_dir = data / 'val'
    val_ld = DataLoader(CaptchaDS(val_dir), BATCH, False, collate_fn=pad, num_workers=2) if val_dir.exists() else None
    net, crit = CRNN().to(dev), nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(net.parameters(), LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=3)
    for ep in range(1, epochs + 1):
        net.train(); s = 0.0
        for imgs, tgts, lens, _ in tr_ld:
            imgs, tgts, lens = imgs.to(dev), tgts.to(dev), lens.to(dev)
            out = net(imgs).log_softmax(2)
            T, B, _ = out.size(); inp = torch.full((B,), T, dtype=torch.long, device=dev)
            loss = crit(out, tgts, inp, lens)
            opt.zero_grad(); loss.backward(); opt.step(); s += loss.item()
        tr = s / len(tr_ld)
        if val_ld:
            vl, acc = eval_loop(net, val_ld, crit, dev)
            print(f"Ep {ep}/{epochs} train={tr:.3f} val={vl:.3f} acc={acc:.2%}"); sched.step(vl)
        else:
            print(f"Ep {ep}/{epochs} train={tr:.3f}"); sched.step(tr)
    torch.save(net.state_dict(), sav)

# ---------- predict ---------------

def predict(model: Path, pattern: str):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:', dev)
    net = CRNN().to(dev)
    state = torch.load(model, map_location=dev, weights_only=True)
    net.load_state_dict(state); net.eval()
    for p in sorted(Path().glob(pattern)):
        img = preprocess(p).unsqueeze(0).to(dev)
        with torch.no_grad():
            print(f"{p.name:<20s} -> {decode(net(img))}")

# ---------- main ------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); subs = parser.add_subparsers(dest='cmd', required=True)
    tr = subs.add_parser('train'); tr.add_argument('--data', type=Path, required=True); tr.add_argument('--epochs', type=int, default=30); tr.add_argument('--model', type=Path, default=Path('model.pth'))
    pr = subs.add_parser('predict'); pr.add_argument('--model', type=Path, required=True); pr.add_argument('--imgs', type=str, default='captchas/*.png')
    args = parser.parse_args()
    if args.cmd == 'train':
        train(args.data, args.epochs, args.model)
    else:
        predict(args.model, args.imgs)
