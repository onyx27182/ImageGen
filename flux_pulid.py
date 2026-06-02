import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import ctypes
import torch
from PIL import Image
from io import BytesIO
import base64

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from diffusers import FluxPipeline

import uvicorn
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_ID = "Raspberry-ai/aidmaRealisticSkin-FLUX-v0.1"

class FluxGenerator:

    def __init__(self):

       if "HF_TOKEN" not in os.environ:
            print("HF_TOKEN is not set.")
            raise SystemExit(1)

       if "API_KEY" not in os.environ:
            print("API_KEY is not set.")
            raise SystemExit(1)

       if not torch.cuda.is_available():
            print("CUDA is not available.")
            raise SystemExit(1)

       self.device = "cuda"
       self.dtype = torch.bfloat16

       print("GPU:", torch.cuda.get_device_name(0))
       print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))

       self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            token=os.environ["HF_TOKEN"],
        ).to(self.device)

       self.pipe.vae.enable_slicing()
       self.pipe.vae.enable_tiling()

       self.pipe.load_lora_weights(
           LORA_ID,
           weight_name="aidmaRealisticSkin-FLUX-v0.1.safetensors",
           adapter_name="skin",
           token=os.environ["HF_TOKEN"],
       )

       self.pipe.set_adapters(["skin"], adapter_weights=[0.35])




    def generate(self, prompt: str) -> Image.Image:
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                prompt_2=prompt,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=5.0,
            ).images[0]

        return result

    def cleanup(self):
        del self.pipe

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        ctypes.CDLL("libc.so.6").malloc_trim(0)



class DolphinRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512

class GenerateRequest(BaseModel):
    prompt: str



app = FastAPI()
flux = FluxGenerator()


@app.get("/")
def health():
    return {"status": "Server is Running for generator"}

@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401)

    # ref = Image.open(BytesIO(base64.b64decode(req.reference_image))).convert("RGB")
    result = flux.generate(prompt=req.prompt)

    buffer = BytesIO()
    result.save(buffer, format="PNG")
    return {"image": base64.b64encode(buffer.getvalue()).decode()}


#-----------------------------     DOLPHIN PROMPT --------------------
@app.post("/dolphin")
def doDolphinPrompt(req: DolphinRequest, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401)
    try:
        response = dolphin.generate(prompt=req.prompt, max_new_tokens=req.max_new_tokens)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

#-----------------------------------------------------------------------





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
