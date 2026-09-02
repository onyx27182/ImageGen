import os
import sys
import gc

import numpy as np
import torch
from PIL import Image

from esrgan_mgr import ESRGANMgr

_PULID_PATH = os.path.expanduser("~/PuLID-FLUX")
if _PULID_PATH not in sys.path:
    sys.path.insert(0, _PULID_PATH)

from flux.sampling import denoise, get_noise, get_schedule, unpack
from stage2_processor import make_img_ids

REALESRGAN_PATH = os.path.expanduser("~/models/RealESRGAN_x4plus.pth")

DEVICE = "cuda"
DTYPE  = torch.bfloat16


def _refine_timesteps(full_steps, image_seq_len, strength):
    # img2img / SDEdit, same convention as diffusers' Flux img2img: build the
    # full shifted schedule, then keep only its tail. `strength` is the fraction
    # of the trajectory that is actually re-noised and re-denoised, so the number
    # of Euler steps run is round(full_steps * strength).
    sched = get_schedule(full_steps, image_seq_len, shift=True)
    init  = max(1, min(full_steps, round(full_steps * strength)))
    return sched[full_steps - init:]


def _spans(total, size, overlap):
    if size >= total:
        return [0]
    step = max(1, size - overlap)
    pos = list(range(0, total - size + 1, step))
    if pos[-1] != total - size:
        pos.append(total - size)
    return pos


def _window(w, h, overlap, left, right, top, bottom):
    # Linear feather on every edge shared with a neighbouring tile; edges that
    # sit on the image border keep weight 1 so the outer frame is untouched.
    o = max(1, min(overlap, w // 2, h // 2))
    ramp = np.arange(1, o + 1, dtype=np.float32) / (o + 1)
    wx = np.ones(w, dtype=np.float32)
    wy = np.ones(h, dtype=np.float32)
    if left:
        wx[:o] = np.minimum(wx[:o], ramp)
    if right:
        wx[-o:] = np.minimum(wx[-o:], ramp[::-1])
    if top:
        wy[:o] = np.minimum(wy[:o], ramp)
    if bottom:
        wy[-o:] = np.minimum(wy[-o:], ramp[::-1])
    return (wy[:, None] * wx[None, :])[:, :, None]


class Stage3Processor:
    def __init__(self):
        self.mgr = ESRGANMgr(REALESRGAN_PATH)
        print("[Stage3] Ready.")

    def process(self, init_image: Image.Image, embeddings: dict,
                flux_model=None, ae=None) -> Image.Image:
        upscale  = int(embeddings.get("upscale", 2))
        target_h = int(embeddings.get("target_height", 0))
        target_w = int(embeddings.get("target_width", 0))

        image = init_image
        if upscale != 0:
            image = self.mgr.run(image, outscale=upscale)

        if target_h and target_w and image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.LANCZOS)

        strength = float(embeddings.get("refine_denoise", 0.0))
        if strength > 0.0 and flux_model is not None and ae is not None:
            image = self._refine(image, embeddings, flux_model, ae, strength)

        return image

    # ── tiled diffusion refine ────────────────────────────────────────────────
    def _refine(self, image, embeddings, flux_model, ae, strength):
        steps     = max(1, int(embeddings.get("refine_steps", 16)))
        guidance  = float(embeddings.get("refine_guidance", 3.0))
        tile      = max(16, (int(embeddings.get("refine_tile_size", 1024)) // 16) * 16)
        id_weight = float(embeddings.get("refine_pulid_weight", 0.0))
        base_seed = int(embeddings.get("seed", 0))

        W, H = image.size
        tw = min(tile, (W // 16) * 16)
        th = min(tile, (H // 16) * 16)
        if tw < 16 or th < 16:
            print(f"[Stage3] refine skipped — image {W}x{H} smaller than one tile")
            return image

        overlap = max(0, min(int(embeddings.get("refine_tile_overlap", 96)),
                             tw // 2, th // 2))

        txt     = embeddings["txt"].to(DEVICE, dtype=DTYPE)
        vec     = embeddings["vec"].to(DEVICE, dtype=DTYPE)
        txt_ids = embeddings["txt_ids"].to(DEVICE)
        if txt_ids.ndim == 2:
            txt_ids = txt_ids.unsqueeze(0)

        id_emb = embeddings.get("id_embeddings")
        use_id = (
            id_weight > 0.0
            and id_emb is not None
            and getattr(flux_model, "pulid_ca", None) is not None
        )
        id_emb = id_emb.to(DEVICE, dtype=DTYPE) if use_id else None

        xs = _spans(W, tw, overlap)
        ys = _spans(H, th, overlap)
        actual_steps = len(_refine_timesteps(steps, (th // 16) * (tw // 16), strength)) - 1
        print(f"[Stage3] Refining {W}x{H} in {len(xs) * len(ys)} tiles "
              f"({tw}x{th}, denoise={strength}, {actual_steps} steps/tile)")

        acc  = np.zeros((H, W, 3), dtype=np.float32)
        wsum = np.zeros((H, W, 1), dtype=np.float32)

        idx = 0
        for y in ys:
            for x in xs:
                crop = image.crop((x, y, x + tw, y + th))
                out  = self._refine_tile(crop, flux_model, ae, txt, txt_ids, vec,
                                         id_emb, id_weight, strength, steps,
                                         guidance, base_seed + idx)
                win = _window(tw, th, overlap,
                              left=x > 0, right=x + tw < W,
                              top=y > 0, bottom=y + th < H)
                acc[y:y + th, x:x + tw]  += np.asarray(out, dtype=np.float32) * win
                wsum[y:y + th, x:x + tw] += win
                idx += 1

        result = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype(np.uint8)

        del acc, wsum, txt, vec, txt_ids, id_emb
        gc.collect()
        torch.cuda.empty_cache()
        return Image.fromarray(result)

    def _refine_tile(self, crop, flux_model, ae, txt, txt_ids, vec,
                     id_emb, id_weight, strength, steps, guidance, seed):
        w, h = crop.size
        arr = np.asarray(crop.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        x_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=DTYPE)

        with torch.inference_mode():
            prev_sample = ae.reg.sample
            ae.reg.sample = False
            try:
                with torch.autocast(device_type="cuda", dtype=DTYPE):
                    z0 = ae.encode(x_img)
            finally:
                ae.reg.sample = prev_sample
            z0 = z0.to(DTYPE)

            seq_len = (h // 16) * (w // 16)
            timesteps = _refine_timesteps(steps, seq_len, strength)
            t0 = timesteps[0]

            noise = get_noise(1, h, w, device=DEVICE, dtype=DTYPE, seed=seed)
            zt = (1.0 - t0) * z0 + t0 * noise           # rectified-flow interpolation at t0
            img, img_ids = make_img_ids(zt)

            out = denoise(
                flux_model,
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                vec=vec,
                timesteps=timesteps,
                guidance=guidance,
                id=id_emb,
                id_weight=id_weight,
                start_step=0,
                true_cfg=1.0,
            )

            out = unpack(out.float(), h, w)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = ae.decode(out)

        out = out.clamp(-1, 1)[0]
        out = ((out + 1.0) * 127.5).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

        del x_img, z0, noise, zt, img, img_ids
        return Image.fromarray(out)
