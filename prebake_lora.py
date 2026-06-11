import os
import torch
from diffusers import QwenImageEditPlusPipeline

project_root = os.path.dirname(os.path.abspath(__file__))
model_dir    = os.path.join(project_root, "models/models/qwen_image_edit")
lora_dir     = os.path.join(project_root, "models")
lora_file    = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
output_dir   = os.path.join(project_root, "models/models/qwen_angles_fused")

print("Loading base model...")
pipe = QwenImageEditPlusPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

print("Loading LoRA weights...")
pipe.load_lora_weights(lora_dir, weight_name=lora_file)

print("Fusing LoRA into base model...")
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

print(f"Saving fused model to {output_dir} ...")
pipe.save_pretrained(output_dir)

print("Done. Update qwen_angles.py to point at qwen_angles_fused and remove the load_lora_weights block.")
