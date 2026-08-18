import contextlib, os, io

from PIL import Image
import cv2
import numpy as np


# Rembg Session
_REMBG_SESSION = None

def _load_rembg():
    global _REMBG_SESSION
    try:
        dn = open(os.devnull, "w")
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            from rembg import new_session
            _REMBG_SESSION = new_session("u2net")
        dn.close()
        print("✓  rembg ready (AI background removal active)", flush=True)
    except Exception as e:
        print(f"i  rembg not available ({type(e).__name__}) — GrabCut fallback active",
              flush=True)

_load_rembg()


# GrabCut Method
def _grabcut(img: Image.Image, bg_color: tuple) -> Image.Image:
    cv_img = np.array(img.convert("RGB"))
    h, w   = cv_img.shape[:2]
    mask   = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mx, my = w // 12, h // 12
    try:
        cv2.grabCut(cv_img, mask, (mx, my, w - 2*mx, h - 2*my),
                    bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return img.convert("RGB")
    mask2  = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    ker    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask2  = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, ker)
    mask2  = cv2.GaussianBlur(mask2, (5, 5), 0)
    r, g, b, _ = img.convert("RGBA").split()
    merged = Image.merge("RGBA", (r, g, b, Image.fromarray(mask2)))
    bg     = Image.new("RGBA", merged.size, bg_color + (255,))
    bg.paste(merged, mask=merged.split()[3])
    return bg.convert("RGB")


# Remgb Method
def remove_background(img: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    if _REMBG_SESSION is not None:
        try:
            dn = open(os.devnull, "w")
            with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
                from rembg import remove as _rem
                buf = io.BytesIO()
                img.convert("RGB").save(buf, "PNG")
                out = _rem(buf.getvalue(), session=_REMBG_SESSION)
            dn.close()
            result = Image.open(io.BytesIO(out)).convert("RGBA")
            bg     = Image.new("RGBA", result.size, bg_color + (255,))
            bg.paste(result, mask=result.split()[3])
            return bg.convert("RGB")
        except Exception:
            pass
    print("  [BG] Using GrabCut fallback")
    return _grabcut(img, bg_color)

