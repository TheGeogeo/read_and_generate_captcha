import argparse, math, os, random, string
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import torch  # type: ignore
    DEVICE = "GPU" if torch.cuda.is_available() else "CPU"
except ModuleNotFoundError:
    DEVICE = "CPU"

WIDTH, HEIGHT = 340, 90
FONT_SIZE_RANGE = (40, 64)
FONT_ROTATION_RANGE = (-30, 30)
CIRCLE_COUNT_RANGE = (1, 3)
CIRCLE_RADIUS_RANGE = (20, 40)
LINE_COUNT_RANGE = (1, 3)
LINE_WIDTH_RANGE = (2, 4)
DOT_COUNT_RANGE = (100, 200)
ENABLE_WARP = True
DEFAULT_MIN_LEN, DEFAULT_MAX_LEN = 4, 7

FallbackFontType = Union[Path, ImageFont.ImageFont]


def _rand_text(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _rand_color(lo: int = 30, hi: int = 200) -> Tuple[int, int, int]:
    return random.randint(lo, hi), random.randint(lo, hi), random.randint(lo, hi)


def _system_fonts() -> List[Path]:
    if os.name == "nt":
        return list((Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts").glob("*.ttf"))[:20]
    return list(Path("/usr/share/fonts").rglob("*.ttf"))[:20] + list(Path("/Library/Fonts").glob("*.ttf"))[:20]


def _load_fonts(dir_: Path) -> List[FallbackFontType]:
    fonts: List[FallbackFontType] = list(dir_.glob("*.ttf")) + list(dir_.glob("*.otf")) if dir_.exists() else []
    if not fonts:
        try:
            import PIL
            p = Path(os.path.dirname(PIL.__file__)) / "fonts/DejaVuSans.ttf"
            if p.exists():
                fonts.append(p)
        except Exception:
            pass
    if not fonts:
        fonts += _system_fonts()
    return fonts or [ImageFont.load_default()]


def _choose_font(fonts: List[FallbackFontType], size: int) -> ImageFont.ImageFont:
    f = random.choice(fonts)
    return ImageFont.truetype(str(f), size) if isinstance(f, Path) else f


def _render_char(ch: str, font: ImageFont.ImageFont, color: Tuple[int, int, int]) -> Image.Image:
    if hasattr(font, "getbbox"):
        x0, y0, x1, y1 = font.getbbox(ch)
        w, h = x1 - x0, y1 - y0
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((-x0, -y0), ch, font=font, fill=color + (255,))
        return im
    w, h = font.getsize(ch)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), ch, font=font, fill=color + (255,))
    return im


def _basic_bg() -> Image.Image:
    bg = Image.new("RGBA", (WIDTH, HEIGHT), _rand_color(190, 230) + (255,))
    d = ImageDraw.Draw(bg)
    for _ in range(random.randint(*DOT_COUNT_RANGE)):
        d.point((random.randint(0, WIDTH), random.randint(0, HEIGHT)), fill=_rand_color(140, 220) + (255,))
    return bg


def _apply_wave(img: Image.Image) -> Image.Image:
    amp, wl = random.uniform(3, 6), random.uniform(120, 200)
    out = Image.new("RGBA", img.size)
    for y in range(img.height):
        shift = int(amp * math.sin(2 * math.pi * y / wl))
        for x in range(img.width):
            nx = x + shift
            if 0 <= nx < img.width:
                out.putpixel((nx, y), img.getpixel((x, y)))
    return out


def _load_bg(dir_: Path) -> Image.Image:
    imgs = list(dir_.glob("*.png")) + list(dir_.glob("*.jpg"))
    if imgs and random.random() < 0.5:
        return Image.open(random.choice(imgs)).convert("RGBA").resize((WIDTH, HEIGHT))
    return _basic_bg()


def generate_captcha(text: Optional[str] = None, length: Optional[int] = None, *, fonts_dir: Union[str, Path] = "fonts", backgrounds_dir: Union[str, Path] = "backgrounds"):
    if text is None:
        text = _rand_text(length or random.randint(DEFAULT_MIN_LEN, DEFAULT_MAX_LEN))
    fonts = _load_fonts(Path(fonts_dir))
    canvas = _load_bg(Path(backgrounds_dir)).copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    x = 15
    for ch in text:
        sz = random.randint(*FONT_SIZE_RANGE)
        font = _choose_font(fonts, sz)
        glyph = _render_char(ch, font, _rand_color())
        glyph = glyph.rotate(random.randint(*FONT_ROTATION_RANGE), expand=True, resample=Image.BICUBIC)
        y = random.randint(5, HEIGHT - glyph.height - 5)
        canvas.alpha_composite(glyph, dest=(x, y))
        x += glyph.width + random.randint(2, 10)
    for _ in range(random.randint(*LINE_COUNT_RANGE)):
        y1, y2 = random.randint(0, HEIGHT), random.randint(0, HEIGHT)
        draw.line([(0, y1), (WIDTH, y2)], fill=_rand_color(60, 180) + (150,), width=random.randint(*LINE_WIDTH_RANGE))
    if ENABLE_WARP:
        canvas = _apply_wave(canvas)
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.6))
    return canvas.convert("RGB"), text


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPTCHA generator (CPU/GPU aware)")
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("--length", type=int)
    ap.add_argument("--text", type=str)
    ap.add_argument("--outdir", type=str, default="generated_captchas")
    ap.add_argument("--fonts-dir", type=str, default="fonts")
    ap.add_argument("--backgrounds-dir", type=str, default="backgrounds")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running on {DEVICE}")
    for i in range(args.count):
        img, txt = generate_captcha(args.text, args.length, fonts_dir=args.fonts_dir, backgrounds_dir=args.backgrounds_dir)
        img.save(out / f"{txt}.png")
        print(f"[{i+1}/{args.count}] saved {txt}.png")


if __name__ == "__main__":
    main()
