import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

model_id = "/home/ubuntu/Dolphin-X1-Trinity-Nano"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": body["prompt"]}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        out = model.generate(input_ids, max_new_tokens=20)
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"response": response}).encode())

HTTPServer(("0.0.0.0", 9998), Handler).serve_forever()
