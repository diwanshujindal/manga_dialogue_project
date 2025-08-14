import argparse
from pathlib import Path
import cv2
import sys

from .utils import derive_out_paths
from .ocr import ocr_with_boxes, boxes_to_mask
from .bubble_detect import detect_text_regions
from .inpaint import inpaint_opencv, inpaint_sd

def build_mask(img, lang: str = "eng", dilate: int = 4):
    text, boxes = ocr_with_boxes(img, lang=lang)
    mask_ocr = boxes_to_mask(img, boxes, dilation=dilate)
    mask_heur = detect_text_regions(img)
    # Combine masks (union)
    mask = cv2.bitwise_or(mask_heur, mask_ocr)
    return text, mask

def run(args):
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}", file=sys.stderr)
        sys.exit(1)

    out_txt, out_mask, out_clean = derive_out_paths(str(img_path), args.outdir)

    if args.mode in ("extract", "both"):
        text, mask = build_mask(img, lang=args.lang, dilate=args.dilate)
        out_txt.write_text(text, encoding="utf-8")
        cv2.imwrite(str(out_mask), mask)
        print(f"[OK] Extracted text -> {out_txt}")
        print(f"[OK] Saved mask -> {out_mask}")
        if args.mode == "extract":
            return

    if args.mode in ("clean", "both"):
        # Build mask if not already produced in this run
        if not out_mask.exists():
            _, mask = build_mask(img, lang=args.lang, dilate=args.dilate)
            cv2.imwrite(str(out_mask), mask)
            print(f"[OK] Saved mask -> {out_mask}")
        else:
            mask = cv2.imread(str(out_mask), cv2.IMREAD_GRAYSCALE)

        if args.method == "opencv":
            clean = inpaint_opencv(img, mask, radius=args.radius, method=args.cv_method)
        elif args.method == "sd":
            clean = inpaint_sd(img, mask, model_id=args.sd_model, prompt=args.prompt)
        else:
            print(f"Unknown method: {args.method}", file=sys.stderr)
            sys.exit(2)

        cv2.imwrite(str(out_clean), clean)
        print(f"[OK] Saved cleaned image -> {out_clean}")

def parse_args():
    p = argparse.ArgumentParser(description="Manga dialogue extractor and cleaner")
    p.add_argument("--image", required=True, help="Path to input image (jpg/png)")
    p.add_argument("--mode", choices=["extract", "clean", "both"], default="both",
                   help="extract: OCR+mask; clean: inpaint only; both: extract then clean")
    p.add_argument("--lang", default="eng", help="Tesseract language code (e.g., eng, jpn, chi_sim)")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--dilate", type=int, default=4, help="Mask dilation pixels (covers outlines)")

    # Inpainting options
    p.add_argument("--method", choices=["opencv", "sd"], default="opencv", help="Inpainting backend")
    p.add_argument("--radius", type=int, default=3, help="OpenCV inpaint radius")
    p.add_argument("--cv_method", choices=["telea", "ns"], default="telea", help="OpenCV inpaint algorithm")
    p.add_argument("--sd_model", default="stabilityai/stable-diffusion-2-inpainting", help="Diffusers model id")
    p.add_argument("--prompt", default=None, help="Optional SD inpainting prompt")

    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
