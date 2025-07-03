# CAPTCHA CRNN – README

A small Python application that trains a **C**onvolutional **R**ecurrent **N**eural **N**etwork to read warped, colored CAPTCHA images.  
It supports GPU acceleration (CUDA) if a compatible NVIDIA card is available.

---

## 1 . Features
| Feature | Details |
|---------|---------|
| **Flexible filenames** | Any file named `<label>.png`/`.jpg` is parsed; no counter or underscore required. |
| **Dynamic width** | Images are rescaled to 32 px height; width is padded on‑the‑fly per mini‑batch. |
| **Automatic inversion** | Background detection : image is inverted only if letters are lighter than background. |
| **GPU auto‑detect** | Prints `Using device: cuda` or `cpu` at launch. |
| **Validation support** | Put images in `dataset/val/` to get `val_loss` + `val_acc` every epoch. |
| **Greedy CTC decoding** | Fast single‑pass character extraction after the CRNN. |

---

## 2 . Project structure
```
captcha_crnn/
│  captcha_crnn.py        # main script (train / predict)
│  requirements.txt
│
├─ dataset/               # training images
│   AYHKy.png
│   YHJG.png
│   ...
└─ captchas/              # real captchas to decode
    *.png
```

---

## 3 . Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```
*The default requirements install the CUDA 12.1 build of PyTorch.*

---

## 4 . Training
```bash
python captcha_crnn.py train --data dataset --epochs 30 --model model.pth
```

---

## 5 . Prediction
```bash
python captcha_crnn.py predict --model model.pth --imgs "captchas/*.png"
```

---

## 6 . Tips
* Increase `BATCH` or reduce `IMG_H` for speed.  
* Add data augmentation inside `preprocess` for robustness.  
* Edit `ALPHABET` to restrict/extend the character set.

---

## 7 . License
MIT
