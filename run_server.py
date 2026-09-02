import faulthandler
import signal
import threading
faulthandler.register(signal.SIGUSR1)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import base64
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
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
    use_reference: bool = True
    image_b64: str | None = None
    file_hash: str = ""
    prompt: str
    height: int = 768
    width: int = 768
    guidance_scale: float = 4.0
    num_inference_steps: int = 28
    seed: int = 0
    pulid_weight: float = 1.0
    num_start_step: int = 0
    true_cfg: float = 1.0
    refine_denoise: float = 0.20
    refine_steps: int = 16
    refine_guidance: float = 3.0
    refine_pulid_weight: float = 0.0
    refine_tile_size: int = 1024
    refine_tile_overlap: int = 96
    upscale: int = 2

class GenerateResponse(BaseModel):
    status: str
    image: str | None

class DolphinRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512


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


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, x_api_key: str = Header(...)):
    print("starting generate!")
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if req.use_reference:
        if not req.image_b64:
            raise HTTPException(status_code=400, detail="image_b64 required when use_reference=True")
        if not req.file_hash:
            raise HTTPException(status_code=400, detail="file_hash required when use_reference=True")
        try:
            print("decoding image!")
            image_bytes = base64.b64decode(req.image_b64)
            id_image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    else:
        id_image = None

    try:
        with _model_lock:
            _ensure_flux_mode()
            _ensure_stages_loaded()

            print("STARTING STAGE1")
            gen_height = req.height // 2 if (req.height == 1920 and req.width == 1080) else req.height
            gen_width  = req.width  // 2 if (req.height == 1920 and req.width == 1080) else req.width
            embeddings = stage1.process(
                id_image=id_image,
                prompt=req.prompt,
                height=gen_height,
                width=gen_width,
                seed=req.seed,
                file_hash=req.file_hash if req.use_reference else "",
            )
            embeddings["target_height"] = req.height
            embeddings["target_width"]  = req.width

            embeddings["guidance_scale"]      = req.guidance_scale
            embeddings["start_step"]          = req.num_start_step
            embeddings["true_cfg"]            = req.true_cfg
            embeddings["id_weight"]           = req.pulid_weight
            embeddings["num_inference_steps"] = req.num_inference_steps
            embeddings["seed"]                = req.seed

            image = stage2.process(embeddings=embeddings)

            embeddings["refine_denoise"]        = req.refine_denoise
            embeddings["refine_steps"]          = req.refine_steps
            embeddings["refine_guidance"]       = req.refine_guidance
            embeddings["refine_pulid_weight"]   = req.refine_pulid_weight
            embeddings["refine_tile_size"]      = req.refine_tile_size
            embeddings["refine_tile_overlap"]   = req.refine_tile_overlap
            embeddings["upscale"]               = req.upscale

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

    return GenerateResponse(status="ok", image=image_b64)


from fastapi.responses import StreamingResponse
import json

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
