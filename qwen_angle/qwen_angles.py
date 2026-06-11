import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


class QWenAngles:

    _LORA_FILE    = "qwen-image-edit-2511-multiple-angles-lora.safetensors"

    def __init__(self):
        self._pipe = None

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._model_dir = os.path.join(project_root, 'models/models', 'qwen_image_edit')
        self._lora_dir  = os.path.join(project_root, 'models')
        lora_path = os.path.join(self._lora_dir, self._LORA_FILE)

        try:
            print("[INIT] QWenAngles: loading base model...")
            self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                self._model_dir,
                torch_dtype=torch.bfloat16,
            )
            self._pipe.enable_model_cpu_offload()

            if not os.path.exists(lora_path):
                raise FileNotFoundError(
                    f"LoRA weights not found at {lora_path}. "
                    f"Run: hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA "
                    f"qwen-image-edit-2511-multiple-angles-lora.safetensors "
                    f"--local-dir models/"
                )
            else:
                print("[INIT] QWenAngles: loading LoRA weights...")
                self._pipe.load_lora_weights(self._lora_dir, weight_name=self._LORA_FILE)
                print("[INIT] QWenAngles ready.")

        except Exception as e:
            import traceback
            print(f"[ERROR] QWenAngles failed to initialise: {type(e).__name__}: {e}")
            traceback.print_exc()
            self._pipe = None

    def process(self, image_rgb: np.ndarray, prompt: str) -> Image.Image:
        """
        Apply a camera-angle change to the image using the LoRA prompt.

        Prompt format:  <sks> [azimuth] [elevation] [distance]

        Azimuth :  front view | left side view | right side view | back view
        Elevation: eye-level shot | high-angle shot | low-angle shot
        Distance : close-up | medium shot | wide shot

        Examples
        --------
        Slight left  : "<sks> left side view eye-level shot medium shot"
        Slight right : "<sks> right side view eye-level shot medium shot"
        Look up      : "<sks> front view high-angle shot medium shot"
        Look down    : "<sks> front view low-angle shot medium shot"

        Returns the original image unchanged if the pipe is unavailable.
        """
        if self._pipe is None:
            raise RuntimeError("QWenAngles pipe not loaded — check initialisation errors above.")

        pil_in = Image.fromarray(image_rgb)

        result = self._pipe(
            image=pil_in,
            prompt=prompt,
            num_inference_steps=50,
            true_cfg_scale=4.0
        ).images[0]

        return result