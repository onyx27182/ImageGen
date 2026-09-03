# Installation

End-to-end setup for the ImageGen server. Every step is required unless marked
**Optional**. Budget ~150 GB of disk for model weights and a machine with an
NVIDIA A100 80 GB (or equivalent ≥ 48 GB card for the lighter pipelines).

The server expects everything to live under the home directory of the account it
runs as (`~` below). Paths are hard-coded in the processors; if you install
somewhere else you must edit them.

---

## 1. Host prerequisites

| Requirement | Version used | Notes |
|-------------|--------------|-------|
| OS          | Ubuntu 22.04 / 24.04 | any recent glibc Linux |
| GPU         | NVIDIA A100 80 GB    | 1 GPU; models are hot-swapped, not sharded |
| NVIDIA driver | ≥ 545 (tested 610.43.02) | must support CUDA 13 |
| Python      | 3.12                 | 3.11 also works |
| Git + build tools | `build-essential`, `git`, `wget` | needed for some wheels / Apex |

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential git wget
```

You do **not** need a system CUDA toolkit for normal operation — the PyTorch
wheels bundle their own CUDA 13 runtime. A matching `nvcc` is only needed if you
build the **Optional** Apex step.

---

## 2. Get the code

```bash
cd ~
git clone https://github.com/onyx27182/ImageGen.git
cd ImageGen
```

> The clone URL currently committed in `.git/config` contains an embedded GitHub
> token — see the security note at the bottom of this file and rotate it.

---

## 3. Python environment

A virtualenv is recommended but the reference box installs into the user site
(`pip install --user`). Either is fine.

```bash
# optional but recommended
python3.12 -m venv ~/.venvs/imagegen
source ~/.venvs/imagegen/bin/activate

# normal install (curated, pinned)
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

`--extra-index-url` is mandatory: `torch==2.11.0+cu130` / `torchvision==0.26.0+cu130`
are not on PyPI.

**For a byte-exact reproduction** of the known-good environment (same transitive
versions, incl. build/dev tooling) use the full snapshot instead:

```bash
pip install -r requirements.lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

### 3a. Patch `basicsr` (required)

`basicsr==1.4.2` imports a symbol that torchvision removed in 0.17. Fix it in
place:

```bash
python - <<'EOF'
import basicsr.data.degradations as m, pathlib
p = pathlib.Path(m.__file__)
p.write_text(p.read_text().replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms.functional import rgb_to_grayscale"))
print("patched", p)
EOF
```

Verify:

```bash
python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print('esrgan OK')"
```

---

## 4. Vendored PuLID-FLUX (required)

The processors add `~/PuLID-FLUX` to `sys.path` and import `flux`, `pulid` and
`eva_clip` from it. Clone it next to the repo and apply two local patches.

```bash
cd ~
git clone https://github.com/ToTheBeginning/PuLID.git PuLID-FLUX
cd PuLID-FLUX
git checkout 1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa   # pin to the tested commit
```

Apply the patches (hard-code weight paths / relax the CLIP check):

```bash
python - <<'EOF'
import pathlib

c = pathlib.Path("flux/modules/conditioner.py")
c.write_text(c.read_text().replace(
    'self.is_clip = version.startswith("openai")',
    'self.is_clip = "openai" in version or "clip" in version.lower()'))

p = pathlib.Path("pulid/pipeline_flux.py")
t = p.read_text().replace("import gc\n\n", "import gc\nimport os\n")
t = t.replace(
    "create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)",
    "create_model_and_transforms('EVA02-CLIP-L-14-336', "
    "os.path.expanduser(\"~/eva_clip_weights/EVA02_CLIP_L_336_psz14_s6B.pt\"), "
    "force_custom_clip=True)")
p.write_text(t)
print("PuLID-FLUX patched")
EOF
```

Do **not** `pip install` PuLID-FLUX's own `requirements.txt` — it pins ancient
torch/diffusers and will downgrade your working set. It is used as a source tree
only.

---

## 5. Model weights

Install the Hugging Face CLI (already pulled in by `huggingface-hub`) and log in
if you need gated repos:

```bash
hf auth login          # required for black-forest-labs/FLUX.1-dev
```

Download each asset to the **exact** path below. Total ≈ 150 GB.

| # | Purpose | Destination | Source |
|---|---------|-------------|--------|
| 1 | FLUX.1-dev base transformer + VAE (BFL single-file) | `~/FLUX.1-dev/flux1-dev.safetensors`, `~/FLUX.1-dev/ae.safetensors` | `black-forest-labs/FLUX.1-dev` *(gated — accept the licence)* |
| 2 | SRPO checkpoint (BFL single-file, bf16) | `~/SRPO/flux.1-dev-SRPO-BFL-bf16.safetensors` | `rockerBOO/flux.1-dev-SRPO` (BFL-format build of `tencent/SRPO`) |
| 3 | T5-XXL text encoder | `~/xflux_text_encoders/` (whole repo) | `XLabs-AI/xflux_text_encoders` |
| 4 | CLIP-L text encoder | `~/clip-vit-large-patch14/` (whole repo) | `openai/clip-vit-large-patch14` |
| 5 | PuLID-FLUX ID adapter | `~/pulid_weights/pulid_flux_v0.9.1.safetensors` | `guozinan/PuLID` |
| 6 | EVA-CLIP vision (PuLID ID encoder) | `~/eva_clip_weights/EVA02_CLIP_L_336_psz14_s6B.pt` | `QuanSun/EVA-CLIP` |
| 7 | InsightFace antelopev2 | `~/insightface/models/antelopev2/` | `DIAMONIK7777/antelopev2` (or let InsightFace auto-download on first run) |
| 8 | RealESRGAN x4plus | `~/models/RealESRGAN_x4plus.pth` | GitHub `xinntao/Real-ESRGAN` v0.1.0 release |
| 9 | Qwen-Image-Edit-2511 base pipeline | `~/models/models/qwen_image_edit/` (whole repo) | `Qwen/Qwen-Image-Edit-2511` |
| 10 | Qwen multi-angle LoRA | `~/models/qwen-image-edit-2511-multiple-angles-lora.safetensors` | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` |
| 11 | Dolphin prompt-expansion LLM | `~/Dolphin-X1-Trinity-Nano/` (whole repo) | `dphn/Dolphin-X1-Trinity-Nano` |

```bash
# 1 — FLUX.1-dev  (gated)
hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors \
    --local-dir ~/FLUX.1-dev

# 2 — SRPO
hf download rockerBOO/flux.1-dev-SRPO flux.1-dev-SRPO-BFL-bf16.safetensors \
    --local-dir ~/SRPO

# 3 — T5-XXL
hf download XLabs-AI/xflux_text_encoders --local-dir ~/xflux_text_encoders

# 4 — CLIP-L
hf download openai/clip-vit-large-patch14 --local-dir ~/clip-vit-large-patch14

# 5 — PuLID adapter
hf download guozinan/PuLID pulid_flux_v0.9.1.safetensors --local-dir ~/pulid_weights

# 6 — EVA-CLIP
hf download QuanSun/EVA-CLIP EVA02_CLIP_L_336_psz14_s6B.pt --local-dir ~/eva_clip_weights

# 7 — antelopev2
hf download DIAMONIK7777/antelopev2 --local-dir ~/insightface/models/antelopev2

# 8 — RealESRGAN
mkdir -p ~/models && wget -O ~/models/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 9 — Qwen-Image-Edit-2511 base
hf download Qwen/Qwen-Image-Edit-2511 --local-dir ~/models/models/qwen_image_edit

# 10 — Qwen multi-angle LoRA
hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA \
    qwen-image-edit-2511-multiple-angles-lora.safetensors --local-dir ~/models

# 11 — Dolphin LLM
hf download dphn/Dolphin-X1-Trinity-Nano --local-dir ~/Dolphin-X1-Trinity-Nano
```

> **FLUX / SRPO format:** both #1 and #2 must be Black Forest Labs
> reference-format single-file checkpoints (identical 780-key layout). Diffusers-
> format folders will not load.

---

## 6. Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `API_KEY` | **yes** | — | sent by clients as the `X-API-KEY` header |
| `FLUX_CKPT` | no | `~/FLUX.1-dev/flux1-dev.safetensors` | dev / PuLID base transformer |
| `SRPO_CKPT` | no | `~/SRPO/flux.1-dev-SRPO-BFL-bf16.safetensors` | SRPO base transformer |

```bash
export API_KEY=choose-a-long-random-string
```

---

## 7. Run

```bash
cd ~/ImageGen
python run_server.py          # uvicorn on 0.0.0.0:8000
```

All pipelines are lazy-loaded on first request, so startup is fast but the first
call of each type takes a while. Smoke test (the root endpoint needs no key):

```bash
curl -s http://localhost:8000/            # -> {"status":"running"}
```

See `README.md` for the endpoint reference and `POST /generate` parameters.

---

## 8. Optional — pre-bake the Qwen angles LoRA

Fusing the LoRA into the base model removes a 10–15 min per-startup cost. The
full procedure is in `PREBAKE_LORA_INSTRUCTIONS.txt`: run the one-time merge
script (`prebake_lora.py` — write it per those instructions; it is not in the
repo), which writes `models/models/qwen_angles_fused/`, then point
`qwen_angle/qwen_angles.py` at `qwen_angles_fused` and drop the
`load_lora_weights()` call.

---

## 9. Optional — NVIDIA Apex

Present in `requirements.lock.txt` but **not imported by the server**. Only build
it if you add code that needs fused optimizers. Requires a system CUDA toolkit
whose major version matches the torch build (CUDA 13).

```bash
git clone https://github.com/NVIDIA/apex ~/apex && cd ~/apex
pip install -v --disable-pip-version-check --no-build-isolation \
    --no-cache-dir --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

---

## Security note — rotate the committed token

`.git/config` in the reference checkout has the remote URL

```
https://ghp_****@github.com/onyx27182/ImageGen.git
```

An embedded personal-access token in a repo checkout is a credential leak. You
should:

1. Revoke that token at <https://github.com/settings/tokens>.
2. Reset the remote to a clean URL and use a credential helper or SSH:
   ```bash
   git remote set-url origin https://github.com/onyx27182/ImageGen.git
   git config --global credential.helper store   # or use SSH keys
   ```
3. Confirm the token was never committed to history:
   `git log -p -S 'ghp_' -- .git` (it lives in `.git/config`, which is not
   tracked, so a plain checkout is clean — but double-check any scripts).
