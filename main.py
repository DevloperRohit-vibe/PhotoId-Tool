
import os,io,uuid,math,traceback,contextlib,base64
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter, ImageEnhance
import PIL.ExifTags
import numpy as np
import cv2


from features.mmToPixel import mm_to_px
from features.backgroundRemover import remove_background
from features.build_PrintSheet import build_print_sheet


# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "outputs"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_SIZES = {
    "passport": (35, 45),
    "aadhaar":  (35, 45),
    "pan":      (25, 35),
    "visa":     (51, 51),
    "stamp":    (25, 30),
    "custom":   (35, 45),
}


# ════════════════════════════════════════════════════════════════════
#  rembg — load once at startup, silently
# ════════════════════════════════════════════════════════════════════
# _REMBG_SESSION = None

# def _load_rembg():
#     global _REMBG_SESSION
#     try:
#         dn = open(os.devnull, "w")
#         with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
#             from rembg import new_session
#             _REMBG_SESSION = new_session("u2net")
#         dn.close()
#         print("✓  rembg ready (AI background removal active)", flush=True)
#     except Exception as e:
#         print(f"ℹ  rembg not available ({type(e).__name__}) — GrabCut fallback active",
#               flush=True)

# _load_rembg()


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=100, optimize=True)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def b64_to_pil(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data)))


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — EXIF ROTATION  (always applied, no toggle)
# ════════════════════════════════════════════════════════════════════

def fix_exif_rotation(img: Image.Image) -> Image.Image:
    """
    Fix camera EXIF orientation. This is always run regardless of the
    auto-rotate toggle because it's metadata, not an aesthetic choice.
    """
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orient_key = next(
            (k for k, v in PIL.ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orient_key and orient_key in exif:
            rotations = {3: 180, 6: 270, 8: 90}
            deg = rotations.get(exif[orient_key])
            if deg:
                print(f"  [EXIF] Rotating {deg}° from EXIF tag")
                img = img.rotate(deg, expand=True)
    except Exception:
        pass
    return img


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — TILT CORRECTION  (only when auto_rotate is ON)
#
#  Root causes of the original bug:
#    • Busy backgrounds (newspapers, posters) trigger many false-positive
#      eye detections → wrong angle computed (e.g. 60°)
#    • Multiple faces detected → wrong face used as reference
#    • No angle sanity check → any computed angle was applied
#
#  Fixes applied:
#    1. Pick the face whose centre is closest to the image centre
#       (the subject is almost always the centred person, not a
#        face on a poster/newspaper in the background).
#    2. Detect eyes only in the upper 55 % of that face ROI.
#    3. Validate the detected eye pair geometrically:
#         • Both eyes must be in the top 60 % of the face
#         • Their Y-diff must be < 20 % of face height
#         • Their X-separation must be 20–60 % of face width
#       → Any pair that fails is discarded (background false-positive).
#    4. Only rotate for angles in range [2°, 8°].
#       • < 2° : imperceptible — skip
#       • > 8° : almost certainly a wrong detection — skip
#    5. Use expand=False so the canvas size stays constant and the
#       face is not zoomed / clipped.  Fill new corners with the
#       image's average border colour to avoid black edges.
# ════════════════════════════════════════════════════════════════════

def _pick_subject_face(faces, img_w, img_h):
    """Return the face whose centre is closest to the image centre."""
    cx_img, cy_img = img_w / 2, img_h / 2
    best, best_dist = None, float("inf")
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw / 2
        cy = fy + fh / 2
        dist = math.hypot(cx - cx_img, cy - cy_img)
        if dist < best_dist:
            best, best_dist = (fx, fy, fw, fh), dist
    return best


def _validated_eye_angle(face_rect, gray_img):
    """
    Detect eyes inside the face ROI, validate the geometry,
    and return the tilt angle in degrees (or None if unreliable).
    """
    fx, fy, fw, fh = face_rect
    eye_casc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # Only search the upper 55 % of the face (where real eyes are)
    eye_roi_h = int(fh * 0.55)
    roi       = gray_img[fy: fy + eye_roi_h, fx: fx + fw]
    eyes      = eye_casc.detectMultiScale(roi, scaleFactor=1.1,
                                          minNeighbors=6, minSize=(20, 20))
    if len(eyes) < 2:
        return None

    # Sort left → right, keep only first two
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1, ey1, ew1, eh1) = eyes[0]
    (ex2, ey2, ew2, eh2) = eyes[1]

    # Absolute image coordinates of eye centres
    cx1 = fx + ex1 + ew1 // 2;  cy1 = fy + ey1 + eh1 // 2
    cx2 = fx + ex2 + ew2 // 2;  cy2 = fy + ey2 + eh2 // 2

    # ── Geometry validation ──────────────────────────────────────────
    # 1. Both eyes must be within the top 60 % of the face
    max_eye_y = fy + fh * 0.60
    if cy1 > max_eye_y or cy2 > max_eye_y:
        print(f"  [ROTATE] Eye Y out of range — false positive, skip")
        return None

    # 2. Their vertical separation must be small (< 20 % of face height)
    y_diff_ratio = abs(cy2 - cy1) / fh
    if y_diff_ratio > 0.20:
        print(f"  [ROTATE] Eye Y-diff {y_diff_ratio:.2f} > 0.20 — skip")
        return None

    # 3. Horizontal separation must be 20–60 % of face width
    x_sep_ratio  = abs(cx2 - cx1) / fw
    if not (0.20 <= x_sep_ratio <= 0.65):
        print(f"  [ROTATE] Eye X-sep {x_sep_ratio:.2f} not in [0.20, 0.65] — skip")
        return None

    angle = math.degrees(math.atan2(cy2 - cy1, cx2 - cx1))
    print(f"  [ROTATE] Validated eye angle: {angle:.2f}°")
    return angle


def correct_rotation(img: Image.Image) -> Image.Image:
    """
    Tilt correction — only applied if a reliable angle is found.
    Rotation range: 2° – 8°.  Outside this range → no change.
    """
    cv_img = np.array(img.convert("RGB"))
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    ih, iw = cv_img.shape[:2]

    face_casc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_casc.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if not len(faces):
        print("  [ROTATE] No face detected — skip")
        return img

    # Pick the subject face (closest to image centre)
    face = _pick_subject_face(faces, iw, ih)

    angle = _validated_eye_angle(face, gray)
    if angle is None:
        return img

    # Range gate: only correct small, credible tilts
    if abs(angle) < 2.0:
        print(f"  [ROTATE] Angle {angle:.1f}° < 2° threshold — no correction needed")
        return img
    if abs(angle) > 8.0:
        print(f"  [ROTATE] Angle {angle:.1f}° > 8° — likely detection error, skip")
        return img

    # Fill colour from border average (avoids black corners)
    border = np.concatenate([cv_img[0, :], cv_img[-1, :],
                              cv_img[:, 0], cv_img[:, -1]])
    fill   = tuple(int(c) for c in border.mean(axis=0))

    print(f"  [ROTATE] Applying tilt correction: {-angle:.2f}°")
    return img.rotate(-angle, expand=False, resample=Image.BICUBIC,
                      fillcolor=fill)


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — LIGHTING CORRECTION
#
#  Root causes of the original bug:
#    • CLAHE always applied regardless of whether the photo needs it.
#      Input image (mean=131, std=50) is already well-exposed — CLAHE
#      added micro-contrast that made it look over-processed.
#    • No gamma correction → dark photos stayed dark.
#
#  Fix:
#    • Analyse mean luminance and std FIRST using the face region
#      (background white wall / newspaper would skew whole-image stats).
#    • Define three zones:
#        SKIP  : mean 90–175 AND std > 30  → already good, do nothing
#        GENTLE: mean 75–90  OR  std 20–30 → mild CLAHE only (clip=1.2)
#        DARK  : mean < 75                 → gamma lift + CLAHE
#        BRIGHT: mean > 175                → gentle compress + CLAHE
#    • CLAHE clipLimit is much lower (1.0–1.8) than the original 2.0.
# ════════════════════════════════════════════════════════════════════

def _face_region_luminance(gray_img, faces_rects):
    """Return mean luminance of the face region (expanded), or whole image."""
    if faces_rects is not None and len(faces_rects):
        ih, iw = gray_img.shape
        fx, fy, fw, fh = _pick_subject_face(faces_rects, iw, ih)
        pad_y = int(fh * 0.55); pad_x = int(fw * 0.25)
        y1 = max(0, fy - pad_y);  y2 = min(ih, fy + fh + pad_y)
        x1 = max(0, fx - pad_x);  x2 = min(iw, fx + fw + pad_x)
        region = gray_img[y1:y2, x1:x2]
        return float(region.mean()), float(region.std())
    return float(gray_img.mean()), float(gray_img.std())


def auto_enhance(img: Image.Image,
                 brightness: float = 1.0,
                 contrast:   float = 1.0,
                 saturation: float = 1.0) -> Image.Image:
    """
    Adaptive lighting + colour correction pipeline.
    Slider values (brightness/contrast/saturation) are always applied
    regardless of the auto-correct logic.
    """
    rgb    = img.convert("RGB")
    cv_img = np.array(rgb, dtype=np.uint8)
    ih, iw = cv_img.shape[:2]
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # ── Detect subject face for region-based analysis ────────────────
    face_casc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_casc.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    mean_lum, std_lum = _face_region_luminance(gray, faces if len(faces) else None)

    print(f"  [LIGHT] Face-region luminance — mean={mean_lum:.1f}  std={std_lum:.1f}")

    # ── Classify exposure ────────────────────────────────────────────
    if 90 <= mean_lum <= 175 and std_lum >= 30:
        zone = "SKIP"
    elif mean_lum < 75:
        zone = "DARK"
    elif mean_lum > 175:
        zone = "BRIGHT"
    else:
        zone = "GENTLE"   # slightly off but not severely

    print(f"  [LIGHT] Zone: {zone}")

    # ── Apply correction based on zone ───────────────────────────────
    if zone == "DARK":
        # Gamma lift towards target luminance 128
        gamma = math.log(128.0 / 255.0) / math.log(max(mean_lum, 5) / 255.0)
        gamma = max(0.40, min(0.80, gamma))
        lut   = np.array([min(255, int((i / 255.0) ** gamma * 255))
                          for i in range(256)], dtype=np.uint8)
        cv_img = cv2.LUT(cv_img, lut)
        print(f"  [LIGHT] Dark gamma lift: {gamma:.3f}")

        # CLAHE after gamma
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l = clahe.apply(l)
        cv_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    elif zone == "BRIGHT":
        # Gentle gamma compress
        gamma = math.log(128.0 / 255.0) / math.log(max(mean_lum, 5) / 255.0)
        gamma = max(1.10, min(1.60, gamma))
        lut   = np.array([min(255, int((i / 255.0) ** gamma * 255))
                          for i in range(256)], dtype=np.uint8)
        cv_img = cv2.LUT(cv_img, lut)
        print(f"  [LIGHT] Bright gamma compress: {gamma:.3f}")

        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
        l = clahe.apply(l)
        cv_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    elif zone == "GENTLE":
        # Very mild CLAHE only — no gamma
        lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        cv_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        print(f"  [LIGHT] Gentle CLAHE applied")

    else:  # SKIP
        print(f"  [LIGHT] Well-exposed — no lighting correction applied")

    # ── COLOUR: Vibrance + skin-tone protection ───────────────────────
    #
    #  "Vibrance" boosts under-saturated colours more than vivid ones,
    #  so skin tones (already warm/saturated) are protected while dull
    #  clothing, backgrounds, etc. get a gentle pop.
    #
    hsv   = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_norm = s_ch / 255.0

    # Vibrance gain: 0 → +20 pts boost, 255 → +4 pts boost
    vibrance = (1.0 - s_norm) * 20.0 + 4.0
    s_ch = np.clip(s_ch + vibrance, 0, 255)

    # Skin-tone guard — pull back the boost for warm/medium-sat pixels
    # (OpenCV HSV: Hue 0-180, Sat 0-255)
    skin_mask = (
        (h_ch >= 0)   & (h_ch <= 18)   &
        (s_norm >= 0.20) & (s_norm <= 0.72)
    ).astype(np.float32)
    skin_mask = cv2.GaussianBlur(skin_mask, (13, 13), 0)

    # Blend: in skin areas keep 60 % of original saturation
    orig_s = np.array(cv2.split(
        cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    )[1])
    s_ch = s_ch * (1.0 - 0.60 * skin_mask) + orig_s * (0.60 * skin_mask)
    s_ch = np.clip(s_ch, 0, 255)

    hsv_out = cv2.merge([h_ch, s_ch, v_ch]).astype(np.uint8)
    cv_img  = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)

    # Subtle skin warmth (+3R, −2B) — only in detected skin pixels
    ycrcb = cv2.cvtColor(cv_img, cv2.COLOR_RGB2YCrCb)
    sk2   = cv2.inRange(ycrcb,
                        np.array([0,   130, 80],  dtype=np.uint8),
                        np.array([255, 175, 128], dtype=np.uint8))
    ker   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sk2   = cv2.dilate(sk2, ker)
    sf    = sk2.astype(np.float32) / 255.0
    cf    = cv_img.astype(np.float32)
    cf[:, :, 0] = np.clip(cf[:, :, 0] + 3.0 * sf, 0, 255)   # R +3
    cf[:, :, 2] = np.clip(cf[:, :, 2] - 2.0 * sf, 0, 255)   # B −2
    cv_img = cf.astype(np.uint8)

    result = Image.fromarray(cv_img)

    # ── User slider adjustments ───────────────────────────────────────
    if abs(brightness - 1.0) > 0.01:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if abs(contrast - 1.0) > 0.01:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if abs(saturation - 1.0) > 0.01:
        result = ImageEnhance.Color(result).enhance(saturation)

    # ── Sharpening — MUCH gentler than original ───────────────────────
    #  Original: radius=1.2, percent=120 → created noise/grain
    #  Fixed:    radius=0.6, percent=50  → adds presence without noise
    result = result.filter(
        ImageFilter.UnsharpMask(radius=0.6, percent=50, threshold=3)
    )

    return result


# ════════════════════════════════════════════════════════════════════
#  FACE DETECTION  (shared utility)
# ════════════════════════════════════════════════════════════════════

def detect_face(cv_img: np.ndarray):
    """
    Return (x, y, w, h) of the subject face.
    Uses centre-proximity to reject background false positives.
    """
    gray  = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    casc  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = casc.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6,
        minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
    )
    if not len(faces):
        return None
    ih, iw = cv_img.shape[:2]
    return _pick_subject_face(faces, iw, ih)


# ════════════════════════════════════════════════════════════════════
#  SMART CROP  (face-centred, exact ID ratio)
# ════════════════════════════════════════════════════════════════════

def smart_crop(img: Image.Image, target_w_mm: int, target_h_mm: int) -> Image.Image:
    """
    1. Detect the subject face.
    2. Scale so face + headroom fills the target height at ~72 %.
    3. Crop centred on the face.
    4. Resize to exact DPI-correct dimensions.
    """
    tw    = mm_to_px(target_w_mm)
    th    = mm_to_px(target_h_mm)
    ratio = tw / th

    cv_img = np.array(img.convert("RGB"))
    face   = detect_face(cv_img)
    ih, iw = cv_img.shape[:2]

    if face is not None:
        fx, fy, fw, fh = face

        # Region: top-of-head → chin+neck
        head_top  = max(0, fy - int(fh * 0.60))
        chin      = min(ih, fy + fh + int(fh * 0.22))
        region_h  = chin - head_top

        # Scale so region_h fills 72 % of th
        scale  = th / (region_h / 0.72)
        new_w  = max(1, int(iw * scale))
        new_h  = max(1, int(ih * scale))
        img_r  = img.resize((new_w, new_h), Image.LANCZOS)
        cv_r   = np.array(img_r.convert("RGB"))

        new_face = detect_face(cv_r)
        if new_face is not None:
            nx, ny, nw, nh = new_face
            cx = nx + nw // 2

            top_r    = max(0, ny - int(nh * 0.60))
            bot_r    = min(new_h, ny + nh + int(nh * 0.22))
            center_y = (top_r + bot_r) // 2

            crop_l = max(0, cx - tw // 2)
            crop_r = crop_l + tw
            if crop_r > new_w:
                crop_r = new_w
                crop_l = max(0, new_w - tw)

            crop_t = max(0, center_y - th // 2)
            crop_b = crop_t + th
            if crop_b > new_h:
                crop_b = new_h
                crop_t = max(0, new_h - th)

            cropped = img_r.crop((crop_l, crop_t, crop_r, crop_b))
        else:
            cropped = img_r.crop((0, 0, min(tw, new_w), min(th, new_h)))

    else:
        # No face → simple centre crop preserving target ratio
        cur_ratio = iw / ih
        if cur_ratio > ratio:
            nw2  = int(ih * ratio);  left = (iw - nw2) // 2
            cropped = img.crop((left, 0, left + nw2, ih))
        else:
            nh2  = int(iw / ratio);  top  = (ih - nh2) // 2
            cropped = img.crop((0, top, iw, top + nh2))

    return cropped.resize((tw, th), Image.LANCZOS)


# ════════════════════════════════════════════════════════════════════
#  BACKGROUND REMOVAL
# ════════════════════════════════════════════════════════════════════

# def _grabcut(img: Image.Image, bg_color: tuple) -> Image.Image:
#     cv_img = np.array(img.convert("RGB"))
#     h, w   = cv_img.shape[:2]
#     mask   = np.zeros((h, w), np.uint8)
#     bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
#     mx, my = w // 12, h // 12
#     try:
#         cv2.grabCut(cv_img, mask, (mx, my, w - 2*mx, h - 2*my),
#                     bgd, fgd, 7, cv2.GC_INIT_WITH_RECT)
#     except Exception:
#         return img.convert("RGB")
#     mask2  = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
#     ker    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     mask2  = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, ker)
#     mask2  = cv2.GaussianBlur(mask2, (5, 5), 0)
#     r, g, b, _ = img.convert("RGBA").split()
#     merged = Image.merge("RGBA", (r, g, b, Image.fromarray(mask2)))
#     bg     = Image.new("RGBA", merged.size, bg_color + (255,))
#     bg.paste(merged, mask=merged.split()[3])
#     return bg.convert("RGB")


# def remove_background(img: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
#     if _REMBG_SESSION is not None:
#         try:
#             dn = open(os.devnull, "w")
#             with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
#                 from rembg import remove as _rem
#                 buf = io.BytesIO()
#                 img.convert("RGB").save(buf, "PNG")
#                 out = _rem(buf.getvalue(), session=_REMBG_SESSION)
#             dn.close()
#             result = Image.open(io.BytesIO(out)).convert("RGBA")
#             bg     = Image.new("RGBA", result.size, bg_color + (255,))
#             bg.paste(result, mask=result.split()[3])
#             return bg.convert("RGB")
#         except Exception:
#             pass
#     print("  [BG] Using GrabCut fallback")
#     return _grabcut(img, bg_color)


# ════════════════════════════════════════════════════════════════════
#  PRINT SHEET  — every row is individually centred on A4
# ════════════════════════════════════════════════════════════════════

# def build_print_sheet(photos: list, count: int,
#                       photo_w_mm: int, photo_h_mm: int) -> Image.Image:
#     A4_W, A4_H = mm_to_px(210), mm_to_px(297)
#     MARGIN, GAP = mm_to_px(8), mm_to_px(3)
#     pw, ph = mm_to_px(photo_w_mm), mm_to_px(photo_h_mm)

#     usable_w = A4_W - 2 * MARGIN
#     cols     = max(1, (usable_w + GAP) // (pw + GAP))
#     rows     = math.ceil(count / cols)
#     total_h  = rows * ph + (rows - 1) * GAP
#     start_y  = MARGIN

#     sheet = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
#     stamp = photos[0].resize((pw, ph), Image.LANCZOS)
#     idx   = 0

#     for row in range(rows):
#         n        = min(cols, count - idx)
#         row_w    = n * pw + (n - 1) * GAP
#         row_x0   = (A4_W - row_w) // 2    # centre each row individually
#         y        = start_y + row * (ph + GAP)
#         for col in range(n):
#             sheet.paste(stamp, (MARGIN + col * (pw + GAP), y))
#             idx += 1

#     return sheet




def process_image():
    try:
        data = request.get_json(force=True)
        # api_key = request.headers.get("API-KEY")
        if request.headers.get("API-KEY") != os.environ.get("API_KEY"):
            print("  [AUTH] Invalid API key")
            return jsonify({"error": "Invalid API key"}), 401
        print("  [AUTH] API key validated")

        image_data  = data.get("image")
        photo_count = int(data.get("photo_count", 4))
        remove_bg   = data.get("remove_bg", True)
        bg_hex      = data.get("bg_color", "#ffffff").lstrip("#")
        size_preset = data.get("size", "passport")
        brightness  = float(data.get("brightness", 1.0))
        contrast    = float(data.get("contrast",   1.0))
        saturation  = float(data.get("saturation", 1.0))
        auto_rotate = data.get("auto_rotate", True)
        manual_crop = data.get("manual_crop", None)

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        bg_color = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))

        print("\n── New photo ─────────────────────────────────")

        # 1. Load + EXIF fix (always)
        img = fix_exif_rotation(b64_to_pil(image_data)).convert("RGB")
        print(f"  [LOAD] {img.size}")

        # 2. Tilt correction (optional, strict)
        if auto_rotate:
            img = correct_rotation(img)
        else:
            print("  [ROTATE] Disabled")

        # 3. Manual crop
        if manual_crop:
            iw, ih = img.size
            cx = int(manual_crop["x"] * iw);  cy = int(manual_crop["y"] * ih)
            cw = max(1, int(manual_crop["w"] * iw))
            ch = max(1, int(manual_crop["h"] * ih))
            img = img.crop((cx, cy, cx + cw, cy + ch))

        # 4. Background removal
        if remove_bg:
            print("  [BG] Removing background…")
            img = remove_background(img, bg_color=bg_color)

        # 5. Lighting + colour correction
        img = auto_enhance(img, brightness, contrast, saturation)

        # 6. Smart crop to exact ID size
        w_mm, h_mm  = ID_SIZES.get(size_preset, (35, 45))
        final_photo = smart_crop(img, w_mm, h_mm)

        # 7. Build A4 sheet
        sheet = build_print_sheet([final_photo], photo_count, w_mm, h_mm)

        # 8. Encode
        photo_b64 = pil_to_b64(final_photo, "PNG")
        sheet_b64 = pil_to_b64(sheet,       "PNG")

        sid = str(uuid.uuid4())[:8]
        # sheet.save(os.path.join(OUTPUT_FOLDER, f"sheet_{sid}.jpg"),"JPEG", dpi=(300, 300), quality=100)

        print(f"  [DONE] sheet_id={sid}")

        return jsonify({
            "success":       True,
            "preview":       f"data:image/png;base64,{photo_b64}",
            "sheet":         f"data:image/png;base64,{sheet_b64}",
            "sheet_id":      sid,
            "photo_size_mm": [w_mm, h_mm],
            "photo_count":   photo_count,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
