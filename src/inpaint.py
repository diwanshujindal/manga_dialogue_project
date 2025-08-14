from typing import Optional
import cv2
import numpy as np

def inpaint_opencv(img, mask, radius: int = 3, method: str = "telea"):
    algo = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, radius, algo)

def inpaint_sd(img_bgr, mask, model_id: str = "stabilityai/stable-diffusion-2-inpainting", prompt: Optional[str] = None):
    """
    Stable Diffusion inpainting via diffusers (optional, GPU recommended).
    """
    try:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Stable Diffusion deps not installed. Install torch, diffusers, transformers, accelerate.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare images
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_mask = Image.fromarray(mask).convert("RGB")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    prompt_text = prompt or "clean manga art, consistent with surrounding style, remove text and restore art"
    result = pipe(prompt=prompt_text, image=pil_img, mask_image=pil_mask).images[0]
    res_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return res_bgr
