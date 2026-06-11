import faulthandler
import signal
faulthandler.register(signal.SIGUSR1)


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import base64
import uuid
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dolphin.dolphin_mgr import DolphinMgr
from llm_mgr import LLMMgr
from qwen_angle.qwen_angles import QWenAngles
import uvicorn

from stage1_processor import Stage1Processor

# ── startup ────────────────────────────────────────────────────────────────
if "API_KEY" not in os.environ:
    print("API_KEY is not set.")
    raise SystemExit(1)

print("Loading Stage1 processor...")
#@PB stage1 = Stage1Processor()
#Do NOT USE dolphin = DolphinMgr()


#@PB llm_mgr = LLMMgr()
app = FastAPI()

print("Loading Stage2 denoiser...")
#@PBfrom stage2_processor import Stage2Processor
#@PBstage2 = Stage2Processor()

print("Loading Stage3 skin refiner...")
from stage3_processor import Stage3Processor
#@PBstage3 = Stage3Processor()


print("Loading QwenAngles...")
qwen_angles = QWenAngles()


# ── request/response models ────────────────────────────────────────────────
class ChangeViewRequest(BaseModel):
    image_b64: str 
    prompt: str
   




# ── request/response models ────────────────────────────────────────────────
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

#-----------------------------     DOLPHIN PROMPT --------------------
@app.post("/dolphin")
def doDolphinPrompt(req: DolphinRequest, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401)
    try:
        # response = dolphin.generate(prompt=req.prompt, max_new_tokens=req.max_new_tokens)
        response = llm_mgr.do_inference(model_name="dphn/Dolphin-X1-Trinity-Nano", 
                                        user_prompt = req.prompt, max_tokens= req.max_new_tokens )
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------------------------



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
        print("STARTING STAGE1")
        embeddings = stage1.process(
            id_image=id_image,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            seed=req.seed,
            file_hash=req.file_hash if req.use_reference else "",
        )
        
        embeddings["guidance_scale"]=req.guidance_scale
        embeddings["start_step"] = req.num_start_step
        embeddings["true_cfg"] = req.true_cfg
        embeddings["id_weight"] = req.pulid_weight
        embeddings["num_inference_steps"]=req.num_inference_steps
        embeddings["seed"] = req.seed


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage1 failed: {e}")

    try:
        image  = stage2.process(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage2 failed: {e}")

    try:
        embeddings["refine_denoise"] = req.refine_denoise
        embeddings["refine_steps"] = req.refine_steps
        embeddings["refine_guidance"] = req.refine_guidance
        embeddings["refine_pulid_weight"] = req.refine_pulid_weight
        embeddings["refine_tile_size"] = req.refine_tile_size
        embeddings["refine_tile_overlap"] = req.refine_tile_overlap

        image = stage3.process(
           init_image=image,
           embeddings=embeddings,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage3 failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    buf = BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(status="ok", image=image_b64)

@app.post("/change_view", response_model=GenerateResponse)
def generate(req: ChangeViewRequest, x_api_key: str = Header(...)):
    print("changing_view!")
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not req.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 required when use_reference=True")
    
    try:
        print("decoding image!")
        image_bytes = base64.b64decode(req.image_b64)
        id_image = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    try:
        print("STARTING STAGE1")
        image = qwen_angles.process(id_image, req.prompt)
        print("Image created successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage1 failed: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    buf = BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    return GenerateResponse(status="ok", image=image_b64)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)