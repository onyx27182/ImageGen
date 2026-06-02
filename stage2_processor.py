import os
import gc
import sys
import torch
from PIL import Image
from einops import rearrange, repeat

sys.path.insert(0, os.path.expanduser("~/PuLID-FLUX"))

import flux.util as flux_util
flux_util.configs["flux-dev"].ckpt_path = os.path.expanduser("~/FLUX.1-dev/flux1-dev.safetensors")
flux_util.configs["flux-dev"].ae_path   = os.path.expanduser("~/FLUX.1-dev/ae.safetensors")

from flux.util import load_ae
from flux.model import Flux
from flux.sampling import denoise, get_noise, get_schedule, unpack
from pulid.pipeline_flux import PuLIDPipeline
from safetensors.torch import load_file as load_sft

PULID_PATH = os.path.expanduser("~/pulid_weights/pulid_flux_v0.9.1.safetensors")

DEVICE     = "cuda"
DTYPE      = torch.bfloat16
ID_WEIGHT  = 1.0
START_STEP = 0
TRUE_CFG   = 1.0


def make_img_ids(img):
    bs, c, h, w = img.shape
    img = rearrange(
        img,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2,
    )
    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] += torch.arange(h // 2)[:, None]
    img_ids[..., 2] += torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img, img_ids.to(img.device)


class Stage2Processor:

    def __init__(self):
        print("[Stage2] Loading FLUX model...")
        ckpt_path = flux_util.configs["flux-dev"].ckpt_path

        # Init on meta device (zero VRAM), load checkpoint to CPU,
        # then move to GPU — avoids double-GPU allocation (46GB peak → 23GB peak).
        with torch.device("meta"):
            self.model = Flux(flux_util.configs["flux-dev"].params)

        sd = load_sft(ckpt_path, device="cpu")
        self.model.load_state_dict(sd, strict=False, assign=True)
        del sd
        gc.collect()

        self.model = self.model.to(dtype=DTYPE, device=DEVICE)
        self.model.eval()

        print("[Stage2] Loading VAE...")
        self.ae = load_ae("flux-dev", device=DEVICE)
        self.ae.eval()

        print("[Stage2] Loading PuLID weights...")
        self.pulid = PuLIDPipeline(self.model, DEVICE, weight_dtype=DTYPE)
        self.pulid.load_pretrain(pretrain_path=PULID_PATH)

        print("[Stage2] Ready.")

    def process(self, embeddings: dict) -> Image.Image:
        txt           = embeddings["txt"].to(DEVICE, dtype=DTYPE)
        vec           = embeddings["vec"].to(DEVICE, dtype=DTYPE)
        txt_ids       = embeddings["txt_ids"].to(DEVICE)
        raw_id = embeddings["id_embeddings"]
        id_embeddings = raw_id.to(DEVICE, dtype=DTYPE) if raw_id is not None else None
        height        = int(embeddings["height"])
        width         = int(embeddings["width"])
        guidance      = float(embeddings["guidance_scale"])
        num_steps     = int(embeddings["num_inference_steps"])
        seed          = int(embeddings["seed"])
        start_step    = int(embeddings["start_step"])
        true_cfg      = float(embeddings["true_cfg"])
        id_weight     = float(embeddings["id_weight"])

        #---------------- print out values -------
        print("=========================  Stage 2 values ====================================")
        print(f"txt:           shape={txt.shape}, dtype={txt.dtype}")
        print(f"vec:           shape={vec.shape}, dtype={vec.dtype}")
        print(f"txt_ids:       shape={txt_ids.shape}, dtype={txt_ids.dtype}")
        print(f"id_embeddings: {None if id_embeddings is None else f'shape={id_embeddings.shape}, dtype={id_embeddings.dtype}'}")
        print(f"height:        {height}")
        print(f"width:         {width}")
        print(f"guidance:      {guidance}")
        print(f"num_steps:     {num_steps}")
        print(f"seed:          {seed}")
        print(f"start_step:    {start_step}")
        print(f"true_cfg:      {true_cfg}")
        print(f"id_weight:     {id_weight}")
        print("=========================  Stage 2 values ====================================")
        
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0)

        noise = get_noise(1, height, width, device=DEVICE, dtype=DTYPE, seed=seed)
        img, img_ids = make_img_ids(noise)

        inp = {
            "img":     img,
            "img_ids": img_ids,
            "txt":     txt,
            "txt_ids": txt_ids.to(DEVICE),
            "vec":     vec,
        }

        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

        print("[Stage2] Denoising...")
        with torch.inference_mode():
            x = denoise(
                self.model,
                **inp,
                timesteps=timesteps,
                guidance=guidance,
                id=id_embeddings,
                id_weight=id_weight,
                start_step=start_step,
                uncond_id=None,
                true_cfg=true_cfg,
                timestep_to_start_cfg=1,
                neg_txt=None,
                neg_txt_ids=None,
                neg_vec=None,
            )

            print("[Stage2] Decoding...")
            x = unpack(x.float(), height, width)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                x = self.ae.decode(x)

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        del x, img, img_ids, txt, txt_ids, vec, id_embeddings
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[Stage2] returning inage")
        return image