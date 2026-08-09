from PIL import Image
from features.mmToPixel import mm_to_px
import math


def build_print_sheet(photos: list, count: int,
                      photo_w_mm: int, photo_h_mm: int) -> Image.Image:
    A4_W, A4_H = mm_to_px(210), mm_to_px(297)
    MARGIN, GAP = mm_to_px(8), mm_to_px(3)
    pw, ph = mm_to_px(photo_w_mm), mm_to_px(photo_h_mm)

    usable_w = A4_W - 2 * MARGIN
    cols     = max(1, (usable_w + GAP) // (pw + GAP))
    rows     = math.ceil(count / cols)
    total_h  = rows * ph + (rows - 1) * GAP
    start_y  = MARGIN

    sheet = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
    stamp = photos[0].resize((pw, ph), Image.LANCZOS)
    idx   = 0

    for row in range(rows):
        n        = min(cols, count - idx)
        row_w    = n * pw + (n - 1) * GAP
        row_x0   = (A4_W - row_w) // 2    # centre each row individually
        y        = start_y + row * (ph + GAP)
        for col in range(n):
            sheet.paste(stamp, (MARGIN + col * (pw + GAP), y))
            idx += 1

    return sheet

