import faulthandler
import signal
import threading
faulthandler.register(signal.SIGUSR1)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import time
import uuid
import asyncio
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from qwen_angle.qwen_angles import QWenAngles
import uvicorn

# ── startup ────────────────────────────────────────────────────────────────
if "API_KEY" not in os.environ:
    print("API_KEY is not set.")
    raise SystemExit(1)

app = FastAPI()

# All models are lazy-loaded on first use
stage1 = None
stage2 = None
stage3 = None
qwen_angles = None
llm_mgr = None

# ── model hot-swap ─────────────────────────────────────────────────────────
# Only one pipeline runs on GPU at a time.  The lock prevents concurrent
# endpoint calls from racing during a device transfer.
_model_lock  = threading.Lock()
_active_mode = "none"   # "none" = nothing on GPU yet (lazy startup)
                        # "qwen" = QwenAngles on GPU, Stage1/2 on CPU
                        # "flux" = Stage1/2 on GPU, QwenAngles on CPU

def _ensure_qwen_loaded():
    global qwen_angles
    if qwen_angles is not None:
        return
    print("[lazy-load] Loading QwenAngles...")
    qwen_angles = QWenAngles()
    print("[lazy-load] QwenAngles ready.")

def _ensure_llm_loaded():
    global llm_mgr
    if llm_mgr is not None:
        return
    # Import deferred so vLLM doesn't initialise CUDA before other models
    print("[lazy-load] Loading LLM manager...")
    from llm_mgr import LLMMgr
    llm_mgr = LLMMgr()
    print("[lazy-load] LLM manager ready.")

def _ensure_qwen_mode():
    global _active_mode
    if _active_mode == "qwen":
        return
    if _active_mode == "flux":
        print("[swap] moving Stage1/Stage2 → CPU")
        for name in ("stage1", "stage2"):
            mgr = globals().get(name)
            if mgr is not None and hasattr(mgr, "to_cpu"):
                mgr.to_cpu()
        gc.collect()
        torch.cuda.empty_cache()
    _ensure_qwen_loaded()
    qwen_angles.to_gpu()
    _active_mode = "qwen"
    print("[swap] done — QwenAngles on GPU")

def _ensure_flux_mode():
    global _active_mode
    if _active_mode == "flux":
        return
    if qwen_angles is not None:
        print("[swap] moving QwenAngles → CPU")
        qwen_angles.to_cpu()
        gc.collect()
        torch.cuda.empty_cache()
    for name in ("stage1", "stage2"):
        mgr = globals().get(name)
        if mgr is not None and hasattr(mgr, "to_gpu"):
            mgr.to_gpu()
    _active_mode = "flux"
    print("[swap] done — Stage1/Stage2 on GPU")

def _ensure_stages_loaded():
    global stage1, stage2, stage3
    if stage1 is not None:
        return
    # QwenAngles is already on CPU at this point (moved by _ensure_flux_mode)
    from stage1_processor import Stage1Processor
    from stage2_processor import Stage2Processor
    from stage3_processor import Stage3Processor
    print("[lazy-load] Loading Stage1/2/3 on first /generate call...")
    stage1 = Stage1Processor()
    stage2 = Stage2Processor()
    stage3 = Stage3Processor()
    print("[lazy-load] Stage1/2/3 ready.")


# ── request/response models ────────────────────────────────────────────────
class ChangeViewRequest(BaseModel):
    image_b64: str
    prompts: list[str]

class ChangeViewResponse(BaseModel):
    status: str
    images: list[str]

class GenerateRequest(BaseModel):
    # ── the four stage toggles ────────────────────────────────────────────
    # A bare request (just `prompt`) runs SRPO Stage 2 and nothing else.
    #   use_srpo        -> SRPO checkpoint. ALWAYS on unless explicitly false
    #                      (false = vanilla flux1-dev).
    #   use_reference   -> PuLID face conditioning. Off unless true (then
    #                      image_b64 + file_hash are required).
    #   use_refine_step -> Stage 3 SDEdit refine pass. Off unless true.
    #   face_detail     -> Stage 3 face-detail pass. Off unless true.
    #   upscale         -> ESRGAN factor. 1 = none.
    use_srpo: bool = True
    use_reference: bool = False
    use_refine_step: bool = False
    prompt: str
    image_b64: str | None = None
    file_hash: str = ""
    height: int = 1024          # final output size; FLUX is trained at 1024
    width: int = 1024
    guidance_scale: float = 4.0
    num_inference_steps: int = 28
    seed: int = 0
    pulid_weight: float = 1.0
    num_start_step: int = 4     # PuLID doc: ~4 for photoreal, 0-1 for stylised
    true_cfg: float = 1.0
    # Stage 3 — tiled refine (runs on the upscaled image, before the downscale)
    refine_denoise: float = 0.25
    refine_steps: int = 24
    refine_guidance: float = 2.5
    refine_pulid_weight: float = 0.0
    refine_tile_size: int = 1024
    refine_tile_overlap: int = 96
    # ESRGAN factor before refine. 0/1 = refine at native res (single tile).
    # 2-4 = supersample, refine tiled, result RETURNED at gen_size*upscale (no
    # downscale). 4 = the ESRGAN model's native factor and a 4096^2 output from a
    # 1024 request — ~25 refine tiles, several minutes, tens of MB of PNG.
    # >4 is just a blurry cv2 resize on top, so it is rejected.
    upscale: int = Field(default=1, ge=0, le=4)
    # Stage 3 — face-detail pass (ADetailer-style; the LAST diffusion step)
    face_detail: bool = False
    face_denoise: float = 0.40
    face_pulid_weight: float | None = None   # None => 0.5 on reference calls, else 0
    face_pad: float = 0.40
    face_steps: int = 30
    face_guidance: float = 3.0

class GenerateResponse(BaseModel):
    status: str
    image: str | None

class DolphinRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512


# ── long render → redirect-polling ────────────────────────────────────────
# A full render (Stage 2 + Stage 3 refine + face pass) runs for minutes. The
# client makes ONE blocking POST /generate, uses requests' default
# redirect-following, and cannot be changed. Cloudflare (in front of the
# thundercompute subdomain) returns 524 if the origin is silent for ~100 s.
#
# So no single HTTP hop is allowed to run that long. The render starts on a
# background thread and POST /generate returns 303 immediately, pointing at
# /generate/wait/{id}. requests follows it automatically (303 -> GET). That
# endpoint waits up to WAIT_SECONDS for the render, then either returns the
# final response — real 200 with the image, or the real 4xx/5xx if it failed —
# or, if the render is still going, 303s back to itself so requests comes
# round again. Every hop is well under the proxy timeout; the client just sees
# its one POST eventually return the true status and body.
# A wait hop is silent until it answers, so WAIT_SECONDS must sit below BOTH the
# proxy's ~100 s ceiling and any read timeout the client sets. 45 s is safely
# under the ~100 s the client already tolerates today.
WAIT_SECONDS = 45
MAX_HOPS     = 20       # ~15 min ceiling; well clear of requests' 30 max_redirects
RENDER_TTL   = 3600     # forget a finished/abandoned render after this long

# id -> {"event": threading.Event, "image": str|None, "exc": HTTPException|None, "ts": float}
_renders: dict[str, dict] = {}
_renders_lock = threading.Lock()


def _start_render(req: "GenerateRequest") -> str:
    """Kick off a render on a background thread, return its id."""
    render_id = uuid.uuid4().hex
    slot = {"event": threading.Event(), "image": None, "exc": None, "ts": time.time()}

    def _work():
        try:
            slot["image"] = _run_generate(req)
        except HTTPException as e:
            slot["exc"] = e
        except Exception as e:                       # pragma: no cover - defensive
            slot["exc"] = HTTPException(status_code=500, detail=f"generate failed: {e}")
        finally:
            slot["event"].set()

    with _renders_lock:
        cutoff = time.time() - RENDER_TTL
        for dead in [rid for rid, s in _renders.items() if s["ts"] < cutoff]:
            del _renders[dead]
        _renders[render_id] = slot

    threading.Thread(target=_work, name="generate", daemon=True).start()
    return render_id


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "running"}

@app.post("/dolphin")
def doDolphinPrompt(req: DolphinRequest, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401)
    _ensure_llm_loaded()
    try:
        response = llm_mgr.do_inference(
            model_name="dphn/Dolphin-X1-Trinity-Nano",
            user_prompt=req.prompt,
            max_tokens=req.max_new_tokens,
        )
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _validate_generate(req: GenerateRequest) -> None:
    """Cheap request validation, run synchronously before the render thread is
    started so an obviously bad request still gets an immediate 4xx."""
    if not req.use_reference:
        return
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 required when use_reference=True")
    if not req.file_hash:
        raise HTTPException(status_code=400, detail="file_hash required when use_reference=True")
    try:
        Image.open(BytesIO(base64.b64decode(req.image_b64))).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _run_generate(req: GenerateRequest) -> str:
    """The actual pipeline. Runs on a background thread. Returns a base64 PNG.
    Raises HTTPException for caller-visible failures, like the old endpoint did."""
    print("starting generate!")

    if req.use_reference:
        try:
            print("decoding image!")
            image_bytes = base64.b64decode(req.image_b64)
            id_image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    else:
        id_image = None

    # ── resolve the stage toggles ─────────────────────────────────────────
    # A bare request runs SRPO Stage 2 only. Each extra stage is opt-in.
    #   use_srpo:  SRPO unless explicitly false (then vanilla flux1-dev; PuLID
    #              was trained on dev, so a reference call gets slightly stronger
    #              likeness with use_srpo=false).
    pulid_used = bool(req.use_reference)
    use_srpo   = bool(req.use_srpo)
    use_refine = bool(req.use_refine_step)
    face_pw    = (0.5 if pulid_used else 0.0) if req.face_pulid_weight is None \
                 else float(req.face_pulid_weight)
    print(f"[generate] pulid_used={pulid_used}  "
          f"use_srpo={use_srpo} (flag={req.use_srpo})  "
          f"use_refine={use_refine} (flag={req.use_refine_step})  "
          f"face_detail={req.face_detail} face_pulid_weight={face_pw}")

    try:
        with _model_lock:
            _ensure_flux_mode()
            _ensure_stages_loaded()

            print("STARTING STAGE1")
            is_hd = (req.height == 1920 and req.width == 1080)
            gen_height = req.height // 2 if is_hd else req.height
            gen_width  = req.width  // 2 if is_hd else req.width
            # HD is generated at half size, so it must be upscaled to reach target
            upscale = max(req.upscale, 2) if is_hd else req.upscale
            embeddings = stage1.process(
                id_image=id_image,
                prompt=req.prompt,
                height=gen_height,
                width=gen_width,
                seed=req.seed,
                file_hash=req.file_hash if req.use_reference else "",
            )
            # Output size. When upscale >= 2 the image is ESRGAN-supersampled and
            # then tiled-refined at that larger size; return it AT that size —
            # downscaling back to the request dims would throw the refine detail
            # straight back out. (For HD, gen_* * upscale already == the request.)
            if upscale >= 2:
                embeddings["target_height"] = gen_height * upscale
                embeddings["target_width"]  = gen_width  * upscale
            else:
                embeddings["target_height"] = req.height
                embeddings["target_width"]  = req.width

            embeddings["guidance_scale"]      = req.guidance_scale
            embeddings["start_step"]          = req.num_start_step
            embeddings["true_cfg"]            = req.true_cfg
            embeddings["id_weight"]           = req.pulid_weight
            embeddings["num_inference_steps"] = req.num_inference_steps
            embeddings["seed"]                = req.seed
            embeddings["use_srpo"]            = use_srpo

            image = stage2.process(embeddings=embeddings)

            embeddings["do_refine"]            = use_refine
            embeddings["refine_denoise"]        = (
                req.refine_denoise if req.refine_denoise > 0.0 else 0.20
            ) if use_refine else req.refine_denoise
            embeddings["refine_steps"]          = req.refine_steps
            embeddings["refine_guidance"]       = req.refine_guidance
            embeddings["refine_pulid_weight"]   = req.refine_pulid_weight
            embeddings["refine_tile_size"]      = req.refine_tile_size
            embeddings["refine_tile_overlap"]   = req.refine_tile_overlap
            embeddings["upscale"]               = upscale

            embeddings["face_detail"]           = req.face_detail
            embeddings["face_denoise"]          = req.face_denoise
            embeddings["face_pulid_weight"]     = face_pw
            embeddings["face_pad"]              = req.face_pad
            embeddings["face_steps"]            = req.face_steps
            embeddings["face_guidance"]         = req.face_guidance

            image = stage3.process(
                init_image=image,
                embeddings=embeddings,
                flux_model=stage2.model,
                ae=stage2.ae,
            )
            del embeddings

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")

    try:
        gc.collect()
        torch.cuda.empty_cache()

        buf = BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"post-process failed: {e}")

    return image_b64


@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str = Header(...)):
    """Start the render and hand the client straight to /generate/wait via 303.

    The client's `requests` call follows the redirect automatically, so from its
    side this is still one blocking POST that returns `{status, image}` (or a
    real error status). The redirect just keeps each HTTP hop short enough to
    clear the proxy's ~100 s timeout.
    """
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    _validate_generate(req)
    render_id = _start_render(req)
    print(f"[generate] render {render_id} started")
    return RedirectResponse(url=f"/generate/wait/{render_id}?hop=1", status_code=303)


@app.get("/generate/wait/{render_id}", response_model=GenerateResponse)
async def generate_wait(render_id: str, hop: int = 1, x_api_key: str = Header(...)):
    """Long-poll one hop of an in-flight render. Returns the final response when
    it is ready, or 303s back to itself while it is still running."""
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    with _renders_lock:
        slot = _renders.get(render_id)
    if slot is None:
        raise HTTPException(status_code=404, detail="Unknown or expired render id")

    deadline = time.monotonic() + WAIT_SECONDS
    while not slot["event"].is_set():
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.5)

    if slot["event"].is_set():
        with _renders_lock:
            _renders.pop(render_id, None)
        if slot["exc"] is not None:
            raise slot["exc"]                        # real 4xx / 5xx, with detail
        return GenerateResponse(status="ok", image=slot["image"])

    if hop >= MAX_HOPS:
        with _renders_lock:
            _renders.pop(render_id, None)
        raise HTTPException(status_code=504,
                            detail=f"render still running after {hop} hops — giving up")
    return RedirectResponse(url=f"/generate/wait/{render_id}?hop={hop + 1}", status_code=303)


@app.post("/change_view")
def change_view(req: ChangeViewRequest, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not req.prompts:
        raise HTTPException(status_code=400, detail="prompts list is empty")
    try:
        image_bytes = base64.b64decode(req.image_b64)
        id_image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    def generate():  # sync generator, not async
        try:
            with _model_lock:
                _ensure_qwen_mode()
                for i, prompt in enumerate(req.prompts):
                    print(f"[change_view] generating {i+1}/{len(req.prompts)}: {prompt}")
                    result = qwen_angles.process(id_image, prompt)
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    yield json.dumps({"status": "ok", "image": b64}) + "\n"
        except Exception as e:
            yield json.dumps({"status": "error", "detail": str(e)}) + "\n"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
