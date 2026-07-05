# structedit/powerpaint_editor.py
import os
import sys
from typing import Optional, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageFilter


class PowerPaintFillEditor:
    """
    Adapter for open-mmlab/PowerPaint v2 / v2-1.

    Expected local layout:
      powerpaint_repo_dir/
        app.py
        powerpaint/...
      checkpoint_dir/
        PowerPaint_Brushnet/
        realisticVisionV60B1_v51VAE/

    edit() is intentionally compatible with your existing FluxFillEditor call.
    """

    def __init__(
        self,
        powerpaint_repo_dir: str,
        checkpoint_dir: str,
        version: str = "ppt-v2",
        weight_dtype: str = "float16",
        guidance_scale: float = 12.0,
        num_inference_steps: int = 45,
        seed: int = 42,
        fitting_degree: float = 1.0,
        local_files_only: bool = True,
        preserve_unmasked: bool = True,
        mask_blur_px: int = 3,
    ):
        self.powerpaint_repo_dir = os.path.abspath(powerpaint_repo_dir)
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.version = version
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.fitting_degree = fitting_degree
        self.local_files_only = local_files_only
        self.preserve_unmasked = preserve_unmasked
        self.mask_blur_px = mask_blur_px

        if not os.path.exists(os.path.join(self.powerpaint_repo_dir, "app.py")):
            raise FileNotFoundError(
                f"Cannot find PowerPaint app.py under: {self.powerpaint_repo_dir}"
            )

        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError(
                f"PowerPaint checkpoint dir not found: {self.checkpoint_dir}"
            )

        if self.powerpaint_repo_dir not in sys.path:
            sys.path.insert(0, self.powerpaint_repo_dir)

        from app import PowerPaintController  # noqa

        if weight_dtype == "float16":
            torch_dtype = torch.float16
        elif weight_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.controller = PowerPaintController(
            weight_dtype=torch_dtype,
            checkpoint_dir=self.checkpoint_dir,
            local_files_only=self.local_files_only,
            version=self.version,
        )

    def _load_image_and_mask(self, image_path: str, mask_path: str):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # PowerPaint/SD inpaint convention: white = repaint, black = keep.
        mask = mask.point(lambda p: 255 if p > 127 else 0)

        return image, mask

    def _composite_to_original_size(self, original: Image.Image, generated: Image.Image, mask: Image.Image):
        if generated.size != original.size:
            generated = generated.resize(original.size, Image.BICUBIC)

        if not self.preserve_unmasked:
            return generated

        alpha = mask.convert("L")
        if alpha.size != original.size:
            alpha = alpha.resize(original.size, Image.NEAREST)

        if self.mask_blur_px > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=self.mask_blur_px))

        alpha_np = np.asarray(alpha).astype(np.float32) / 255.0
        alpha_np = alpha_np[..., None]

        original_np = np.asarray(original).astype(np.float32)
        generated_np = np.asarray(generated).astype(np.float32)

        out = generated_np * alpha_np + original_np * (1.0 - alpha_np)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def edit(
        self,
        image_path: str,
        mask_path: str,
        target_prompt: str,
        output_path: str,
        prompt_suffix: str = "",
        task_type: str = "text-guided",
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        image, mask = self._load_image_and_mask(image_path, mask_path)

        if task_type in ["object-removal", "delete", "remove"]:
            task = "object-removal"
            prompt = ""
            negative_prompt = negative_prompt or (
                "object, subject, foreground object, person, animal, text, logo, watermark"
            )
        else:
            task = "text-guided"
            prompt = target_prompt or ""
            if prompt_suffix:
                prompt = f"{prompt}, {prompt_suffix}".strip(", ")
            negative_prompt = negative_prompt or (
                "worst quality, low quality, blurry, distorted, text, watermark, logo"
            )

        input_image = {
            "image": image,
            "mask": mask.convert("RGB"),
        }

        outputs, aux = self.controller.infer(
            input_image=input_image,
            text_guided_prompt=prompt,
            text_guided_negative_prompt=negative_prompt,
            shape_guided_prompt=prompt,
            shape_guided_negative_prompt=negative_prompt,
            fitting_degree=self.fitting_degree,
            ddim_steps=self.num_inference_steps,
            scale=self.guidance_scale,
            seed=self.seed,
            task=task,
            vertical_expansion_ratio=None,
            horizontal_expansion_ratio=None,
            outpaint_prompt="",
            outpaint_negative_prompt=negative_prompt,
            removal_prompt="",
            removal_negative_prompt=negative_prompt,
        )

        if not outputs:
            raise RuntimeError("PowerPaint returned no image.")

        result = outputs[0]
        result = result.convert("RGB")
        result = self._composite_to_original_size(image, result, mask)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)

        return {
            "backend": "powerpaint",
            "task": task,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_path": output_path,
            "checkpoint_dir": self.checkpoint_dir,
            "version": self.version,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }
