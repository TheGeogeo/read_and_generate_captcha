#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement :
    python captcha_ml.py train --data dataset --model svm.joblib

Prédiction :
    python captcha_ml.py predict --model svm.joblib --imgs "captchas/*.png"
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from joblib import dump, load

# —————————————————— Constantes ——————————————————
GLYPH_SIZE = 32  # taille de normalisation avant HOG
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# —————————————————— Helpers ———————————————————

def segment_glyphs(img: np.ndarray) -> List[np.ndarray]:
    """Retourne la liste des glyphes binaires (fond noir, lettres blanches)."""
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes.sort(key=lambda b: b[0])
    return [bw[y:y + h, x:x + w] for x, y, w, h in boxes]


def hog_descriptor(glyph: np.ndarray) -> np.ndarray:
    """Calcule le descripteur HOG 9 bins d’un glyphe 32×32."""
    g = cv2.resize(glyph, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hog = cv2.HOGDescriptor(_winSize=(GLYPH_SIZE, GLYPH_SIZE),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    return hog.compute(g).ravel()

# —————————————————— Dataset ——————————————————

def load_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Charge le dossier *root* et retourne X (HOG) et y (labels char)."""
    X, y = [], []
    for p in sorted(root.glob("*.png")):
        label = p.stem
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        glyphs = segment_glyphs(img)
        if len(glyphs) != len(label):
            continue  # mauvais échantillon (segmentation échouée)
        for g, c in zip(glyphs, label):
            if c not in ALPHABET:
                continue
            X.append(hog_descriptor(g))
            y.append(c)
    return np.array(X), np.array(y)

# —————————————————— Train ———————————————————

def train(data_dir: Path, model_path: Path):
    X, y = load_dataset(data_dir)
    if len(X) == 0:
        raise RuntimeError("Dataset vide ou segmentation impossible.")

    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual=False, max_iter=10000))
    clf.fit(X, y)
    print(classification_report(y, clf.predict(X)))

    dump(clf, model_path)
    print("Model saved to", model_path)

# —————————————————— Predict ——————————————————

def predict(model_path: Path, pattern: str):
    clf = load(model_path)
    for p in sorted(Path().glob(pattern)):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(p.name, "-> ERREUR (lecture)")
            continue
        glyphs = segment_glyphs(img)
        if not glyphs:
            print(p.name, "-> (aucun glyphe)")
            continue
        feats = [hog_descriptor(g) for g in glyphs]
        pred = "".join(clf.predict(feats))
        print(f"{p.name:<20s} -> {pred}")

# —————————————————— CLI ————————————————————
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", type=Path, required=True)
    tr.add_argument("--model", type=Path, default=Path("svm.joblib"))

    pr = sub.add_parser("predict")
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--imgs", type=str, default="captchas/*.png")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.data, args.model)
    else:
        predict(args.model, args.imgs)
