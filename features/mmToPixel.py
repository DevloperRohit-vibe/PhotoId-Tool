
DPI       = 300
MM_PER_IN = 25.4

def mm_to_px(mm: float) -> int:
    return int((mm / MM_PER_IN) * DPI)