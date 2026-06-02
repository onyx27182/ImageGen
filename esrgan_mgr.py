import cv2
import numpy as np
import torch
from PIL import Image

DEVICE = "cuda"


class ESRGANMgr:

    def __init__(self, realesrgan_model_path: str):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=realesrgan_model_path,
            model=rrdb,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=torch.device(DEVICE),
        )
        print("[ESRGANMgr] Ready.")

    def run(self, init_image: Image.Image, outscale: int = 2) -> Image.Image:
        print(f"[ESRGANMgr] Upscaling {outscale}x with RealESRGAN...")
        img_bgr = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
        output_bgr, _ = self.upsampler.enhance(img_bgr, outscale=outscale)
        result = Image.fromarray(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB))
        print(f"[ESRGANMgr] {init_image.size[0]}x{init_image.size[1]} -> {result.size[0]}x{result.size[1]}")
        return result
