import os
from PIL import Image
from esrgan_mgr import ESRGANMgr

REALESRGAN_PATH = os.path.expanduser("~/models/RealESRGAN_x4plus.pth")


class Stage3Processor:
    def __init__(self):
        self.mgr = ESRGANMgr(REALESRGAN_PATH)
        print("[Stage3] Ready.")

    def process(self, init_image: Image.Image, embeddings: dict) -> Image.Image:
        return init_image
