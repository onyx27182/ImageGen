import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from face_processor import FaceProcessor


#@PBos.environ["TRANSFORMERS_OFFLINE"] = "1"
#@PBos.environ["HF_DATASETS_OFFLINE"] = "1"
#TODO mv ~/insightface/models/antelopev2/antelopev2/* ~/insightface/models/antelopev2/
#rmdir ~/insightface/models/antelopev2/antelopev2
import gc
import sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/PuLID-FLUX"))
from flux.sampling import get_noise, prepare
from flux.modules.conditioner import HFEmbedder

DTYPE  = torch.bfloat16
DEVICE = "cuda"

class Stage1Processor:

    def __init__(self):

        # ── load text encoders ─────────────────────────────────────────────
        print("[Stage1] >>> Loading text encoders...")
        self.t5 = HFEmbedder(
            os.path.expanduser("~/xflux_text_encoders"),
            max_length=512,
            torch_dtype=torch.bfloat16,
             ).to(DEVICE)
        print(">>> T5 loaded.....")     

        print("[Stage1] >>> Loading HF embedder...")
        self.clip = HFEmbedder(
            os.path.expanduser("~/clip-vit-large-patch14"),
            max_length=77,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)

        print(">>> HF loaded...")

        self._face_processor = None

        print("[Stage1] Ready.")

    def to_cpu(self):
        self.t5.to("cpu")
        self.clip.to("cpu")
        if self._face_processor is not None:
            fp = self._face_processor
            fp.eva_clip.to("cpu")
            fp.id_encoder.to("cpu")
            fp.face_helper.face_det.to("cpu")
            fp.face_helper.face_parse.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def to_gpu(self):
        self.t5.to(DEVICE)
        self.clip.to(DEVICE)
        if self._face_processor is not None:
            fp = self._face_processor
            fp.eva_clip.to(DEVICE)
            fp.id_encoder.to(DEVICE)
            fp.face_helper.face_det.to(DEVICE)
            fp.face_helper.face_parse.to(DEVICE)

    def process(
        self,
        id_image: np.ndarray | None,
        prompt: str,
        height: int = 768,
        width: int = 768,
        seed: int=0,
        file_hash: str = ""
    ) -> dict:

        # ── encode prompt ──────────────────────────────────────────────────
        print("[Stage1] Encoding prompt...")
        noise = get_noise(1, height, width, device=DEVICE, dtype=DTYPE, seed=seed)
        inp   = prepare(t5=self.t5, clip=self.clip, img=noise, prompt=prompt)

        txt     = inp["txt"]
        txt_ids = inp["txt_ids"]
        vec     = inp["vec"]
        del noise, inp

 
        # ── process face image ─────────────────────────────────────────────
        if id_image is not None:
            if self._face_processor is None:
                print("[Stage1] >>> Loading Face Processor...")
                self._face_processor = FaceProcessor(DEVICE, DTYPE)
            print("[Stage1] Processing face...")
            id_embeddings = self._face_processor.process(id_image, filehash_cache_key=file_hash or None)
        else:
            print("[Stage1] Skipping face processing (use_reference=False)")
            id_embeddings = None

        return {
            "txt":                 txt,
            "txt_ids":             txt_ids,
            "vec":                 vec,
            "id_embeddings":       id_embeddings,
            "height":              height,
            "width":               width,
        }