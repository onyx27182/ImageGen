import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model_id = "/home/ubuntu/Dolphin-X1-Trinity-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

app = FastAPI()

class Req(BaseModel):
    prompt: str

@app.post("/test")
def test(req: Req):
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids, max_new_tokens=20)
    return {"response": tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
