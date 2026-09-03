# Installation — without PuLID (no identity / reference-image support)

Same server, set up **without** the PuLID identity feature. Use this if you will
never send a reference image (`use_reference` / `id_image`). Text-to-image on
both the **dev** and **SRPO** checkpoints, the Stage 3 upscale + tiled refine +
face-detail pass, the Qwen multi-angle `/change_view` endpoint and the
`/dolphin` prompt expander all work normally.

**What is skipped vs. the full [INSTALL.md](INSTALL.md):**

| Skipped | Item |
|---------|------|
| Weight | PuLID-FLUX ID adapter — `pulid_flux_v0.9.1.safetensors` (~1.1 GB) |
| Weight | EVA-CLIP vision encoder — `EVA02_CLIP_L_336_psz14_s6B.pt` (~0.9 GB) |
| Patch  | PuLID-FLUX patch #2 (the hard-coded EVA-CLIP path) |

Everything else is identical — including the Python packages. The processors
still `import` the PuLID pipeline module at load time, so `insightface`,
`facexlib` and the vendored `eva_clip` package must all be importable; only the
**weights** above are unnecessary. If a client does send a reference image the
request will fail with a missing-file error — that is expected for this build.

Budget ~125 GB of disk for weights (vs ~150 GB for the full build). Target
machine: NVIDIA A100 80 GB (or a ≥ 48 GB card).

---

## 1. Host prerequisites

| Requirement | Version used | Notes |
|-------------|--------------|-------|
| OS          | Ubuntu 22.04 / 24.04 | any recent glibc Linux |
| GPU         | NVIDIA A100 80 GB    | 1 GPU; models are hot-swapped, not sharded |
| NVIDIA driver | ≥ 545 (tested 610.43.02) | must support CUDA 13 |
| Python      | 3.12                 | 3.11 also works |
| Git + build tools | `build-essential`, `git`, `wget` | needed for some wheels |

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential git wget
```

A system CUDA toolkit is not needed — the PyTorch wheels bundle their own CUDA 13
runtime.

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

```bash
# optional but recommended
python3.12 -m venv ~/.venvs/imagegen
source ~/.venvs/imagegen/bin/activate

# curated, pinned deps
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

`--extra-index-url` is mandatory: `torch==2.11.0+cu130` /
`torchvision==0.26.0+cu130` are not on PyPI.

For a byte-exact clone of the known-good environment use `requirements.lock.txt`
instead (same command, same `--extra-index-url`).

### 3a. Patch `basicsr` (required)

`basicsr==1.4.2` imports a symbol torchvision removed in 0.17. Fix it in place:

```bash
python - <<'EOF'
import basicsr.data.degradations as m, pathlib
p = pathlib.Path(m.__file__)
p.write_text(p.read_text().replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms.functional import rgb_to_grayscale"))
print("patched", p)
EOF

python -c "from basicsr.archs.rrdbnet_arch import RRDBNet; from realesrgan import RealESRGANer; print('esrgan OK')"
```

---

## 4. Vendored PuLID-FLUX tree (still required)

The processors add `~/PuLID-FLUX` to `sys.path` and import the `flux` package
(sampling, model, VAE, text-encoder wrapper) from it. The `pulid` and `eva_clip`
packages ship in the same tree and must remain importable, but you will not
download their weights.

```bash
cd ~
git clone https://github.com/ToTheBeginning/PuLID.git PuLID-FLUX
cd PuLID-FLUX
git checkout 1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa   # tested commit
```

Apply **only patch #1** — the CLIP-vs-T5 detection fix in the text-encoder
wrapper (needed for the CLIP encoder to load; unrelated to PuLID):

```bash
python - <<'EOF'
import pathlib
c = pathlib.Path("flux/modules/conditioner.py")
c.write_text(c.read_text().replace(
    'self.is_clip = version.startswith("openai")',
    'self.is_clip = "openai" in version or "clip" in version.lower()'))
print("conditioner.py patched")
EOF
```

Skip PuLID-FLUX patch #2 (it only rewrites a path used when the PuLID pipeline is
actually instantiated, which never happens in this build).

Do **not** `pip install` PuLID-FLUX's own `requirements.txt` — it pins ancient
torch/diffusers and would downgrade your working set. It is a source tree only.

---

## 5. Model weights

```bash
hf auth login          # required for black-forest-labs/FLUX.1-dev
```

Download each asset to the **exact** path below. Total ≈ 125 GB.

| # | Purpose | Destination | Source |
|---|---------|-------------|--------|
| 1 | FLUX.1-dev base transformer + VAE (BFL single-file) | `~/FLUX.1-dev/flux1-dev.safetensors`, `~/FLUX.1-dev/ae.safetensors` | `black-forest-labs/FLUX.1-dev` *(gated — accept the licence)* |
| 2 | SRPO checkpoint (BFL single-file, bf16) | `~/SRPO/flux.1-dev-SRPO-BFL-bf16.safetensors` | `rockerBOO/flux.1-dev-SRPO` (BFL-format build of `tencent/SRPO`) |
| 3 | T5-XXL text encoder | `~/xflux_text_encoders/` (whole repo) | `XLabs-AI/xflux_text_encoders` |
| 4 | CLIP-L text encoder | `~/clip-vit-large-patch14/` (whole repo) | `openai/clip-vit-large-patch14` |
| 5 | InsightFace antelopev2 (Stage 3 face-detail pass) | `~/insightface/models/antelopev2/` | `DIAMONIK7777/antelopev2` (or let InsightFace auto-download on first run) |
| 6 | RealESRGAN x4plus (Stage 3 upscale) | `~/models/RealESRGAN_x4plus.pth` | GitHub `xinntao/Real-ESRGAN` v0.1.0 release |
| 7 | Qwen-Image-Edit-2511 base pipeline (`/change_view`) | `~/models/models/qwen_image_edit/` (whole repo) | `Qwen/Qwen-Image-Edit-2511` |
| 8 | Qwen multi-angle LoRA | `~/models/qwen-image-edit-2511-multiple-angles-lora.safetensors` | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` |
| 9 | Dolphin prompt-expansion LLM (`/dolphin`) | `~/Dolphin-X1-Trinity-Nano/` (whole repo) | `dphn/Dolphin-X1-Trinity-Nano` |

```bash
# 1 — FLUX.1-dev  (gated; also supplies the VAE used by every pipeline)
hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors \
    --local-dir ~/FLUX.1-dev

# 2 — SRPO
hf download rockerBOO/flux.1-dev-SRPO flux.1-dev-SRPO-BFL-bf16.safetensors \
    --local-dir ~/SRPO

# 3 — T5-XXL
hf download XLabs-AI/xflux_text_encoders --local-dir ~/xflux_text_encoders

# 4 — CLIP-L
hf download openai/clip-vit-large-patch14 --local-dir ~/clip-vit-large-patch14

# 5 — antelopev2  (Stage 3 face-detail detector)
hf download DIAMONIK7777/antelopev2 --local-dir ~/insightface/models/antelopev2

# 6 — RealESRGAN
mkdir -p ~/models && wget -O ~/models/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 7 — Qwen-Image-Edit-2511 base
hf download Qwen/Qwen-Image-Edit-2511 --local-dir ~/models/models/qwen_image_edit

# 8 — Qwen multi-angle LoRA
hf download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA \
    qwen-image-edit-2511-multiple-angles-lora.safetensors --local-dir ~/models

# 9 — Dolphin LLM
hf download dphn/Dolphin-X1-Trinity-Nano --local-dir ~/Dolphin-X1-Trinity-Nano
```

> **FLUX / SRPO format:** #1 and #2 must be Black Forest Labs reference-format
> single-file checkpoints (identical 780-key layout). Diffusers-format folders
> will not load. `Stage2Processor` loads the **dev** checkpoint at startup and
> swaps to SRPO on demand, so both files must be present even for SRPO-only use.

**Not downloaded in this build:** `~/pulid_weights/` and `~/eva_clip_weights/`.

---

## 6. Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `API_KEY` | **yes** | — | sent by clients as the `X-API-KEY` header |
| `FLUX_CKPT` | no | `~/FLUX.1-dev/flux1-dev.safetensors` | dev base transformer |
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

Pipelines are lazy-loaded on first request. Smoke test (root needs no key):

```bash
curl -s http://localhost:8000/            # -> {"status":"running"}
```

A text-to-image call (no reference image):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "X-API-KEY: $API_KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"a red fox in snow","use_srpo":true}'
```

See `README.md` for the full endpoint and parameter reference. Ignore anything
about reference images / `use_reference` / identity — that path is not installed
here.

---

## 8. Optional — pre-bake the Qwen angles LoRA

Fusing the LoRA into the base model removes a 10–15 min per-startup cost. The
procedure is in `PREBAKE_LORA_INSTRUCTIONS.txt`: run the one-time merge script
(`prebake_lora.py` — write it per those instructions; it is not in the repo),
which writes `models/models/qwen_angles_fused/`, then point
`qwen_angle/qwen_angles.py` at `qwen_angles_fused` and drop the
`load_lora_weights()` call.

---

## Security note — rotate the committed token

`.git/config` in the reference checkout has the remote URL
`https://ghp_****@github.com/onyx27182/ImageGen.git`. An embedded personal-access
token is a credential leak:

1. Revoke it at <https://github.com/settings/tokens>.
2. Reset the remote and use a credential helper or SSH:
   ```bash
   git remote set-url origin https://github.com/onyx27182/ImageGen.git
   git config --global credential.helper store
   ```
