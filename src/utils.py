from pathlib import Path
from typing import Tuple

def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def derive_out_paths(image_path: str, outdir: str) -> Tuple[Path, Path, Path]:
    outdir_path = ensure_outdir(outdir)
    stem = Path(image_path).stem
    txt = outdir_path / f"{stem}_text.txt"
    mask = outdir_path / f"{stem}_mask.png"
    clean = outdir_path / f"{stem}_clean.png"
    return txt, mask, clean
