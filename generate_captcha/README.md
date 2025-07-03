# CAPTCHA Generator

A small Python tool that creates PNG CAPTCHAs inspired by the examples you provided.  
It auto‑detects whether PyTorch with CUDA is available to display **CPU** or **GPU** usage (no GPU required).

---

## 1. Requirements

| Package | Version |
|---------|---------|
| Pillow  | ≥ 9.5 &lt; 11 |
| PyTorch | *(optional)* – only for GPU detection (`pip install torch --index-url https://download.pytorch.org/whl/cu118` or similar) |

Python ≥ 3.8 recommended.

---

## 2. Installation

```bash
# Clone / copy this repo
git clone <your‑repo> captcha-gen
cd captcha-gen

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contains only Pillow.  
Install PyTorch separately if you want the **GPU** detection message.

---

## 3. Directory structure

```
captcha-gen/
├─ captcha_generator.py      # main script
├─ fonts/                    # (optional) custom .ttf/.otf files
├─ backgrounds/              # (optional) background images (png/jpg)
├─ generated_captchas/       # output folder (auto‑created)
└─ requirements.txt
```

---

## 4. Usage

### Basic

```bash
python captcha_generator.py
```

Generates **one** CAPTCHA with 4‑7 random characters into `generated_captchas/`.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --count` | Number of images to create | `1` |
| `--length` | Fixed length for random text | random 4‑7 |
| `--text` | Force specific text (overrides `--length`) | *none* |
| `--outdir` | Output directory | `generated_captchas` |
| `--fonts-dir` | Directory with `.ttf` / `.otf` fonts | `fonts` |
| `--backgrounds-dir` | Directory with background images | `backgrounds` |

Examples:

```bash
# 10 captchas of 6 characters
python captcha_generator.py -n 10 --length 6

# 5 identical captchas with text HELLO5
python captcha_generator.py -n 5 --text HELLO5

# Custom folders
python captcha_generator.py --fonts-dir my_fonts --backgrounds-dir my_bgs
```

During execution the script prints:

```
Running on GPU   # or CPU
[1/5] saved Ab3X7.png
...
```

---

## 5. Customisation tips

* **Fonts:** Drop any `.ttf` / `.otf` into `fonts/`.  
* **Colours:** Tweak `_rand_color()` ranges in the script to change palette.  
* **Noise & distortion:** `DOT_COUNT_RANGE`, `ENABLE_WARP`, etc.  
* **Occlusion logic:** radius and count in `CIRCLE_*` constants.

---

## 6. License

MIT (or change to your preferred license).
