# ImageGen

Multi-model FastAPI image-generation server. One GPU (A100 80GB); only one
pipeline is resident on the GPU at a time and models are hot-swapped between
requests.

## Running

```bash
export API_KEY=...            # required; clients send it as the X-API-KEY header
python run_server.py          # uvicorn on 0.0.0.0:8000
```

Optional environment overrides:

| Variable    | Default                                             | Purpose |
|-------------|----------------------------------------------------|---------|
| `API_KEY`   | *(required)*                                        | Auth token for every endpoint |
| `FLUX_CKPT` | `~/FLUX.1-dev/flux1-dev.safetensors`                | The **dev / PuLID** base transformer |
| `SRPO_CKPT` | `~/SRPO/flux.1-dev-SRPO-BFL-bf16.safetensors`       | The **SRPO** base transformer |

Both checkpoints must be single-file checkpoints in the Black Forest Labs
reference format (identical 780-key layout).

## Endpoints

| Method | Path           | Purpose |
|--------|----------------|---------|
| `GET`  | `/`            | Health check |
| `POST` | `/generate`    | Image generation — Stage 1 (text/face encode) → Stage 2 (FLUX denoise) → Stage 3 (upscale + optional refine) |
| `POST` | `/change_view` | Camera-angle change via QwenImageEdit angles LoRA (NDJSON stream) |
| `POST` | `/dolphin`     | LLM chat (vLLM, lazy-loaded on first call) |

All endpoints require the `X-API-KEY` header.

---

## `POST /generate`

### Request body

| Field                 | Type          | Default | Notes |
|-----------------------|---------------|---------|-------|
| `prompt`              | str           | *(required)* | Text prompt |
| `use_reference`       | bool          | `true`  | Send a face image to preserve identity (PuLID). See below. |
| `image_b64`           | str \| null   | `null`  | Base64 PNG/JPEG of the reference face. Required when `use_reference=true`. |
| `file_hash`           | str           | `""`    | Stable key for the reference image; used to cache the face embedding. Required when `use_reference=true`. |
| `use_srpo`            | bool \| null  | `null`  | Base-checkpoint override. `null` = auto. See [Checkpoint selection](#checkpoint-selection). |
| `use_refine_step`     | bool \| null  | `null`  | Stage 3 refine override. `null` = auto (on). See [The refine step](#the-refine-step). |
| `height`              | int           | `768`   | Output height. `1920x1080` is generated at half size then upscaled. |
| `width`               | int           | `768`   | Output width. |
| `guidance_scale`      | float         | `4.0`   | Stage 2 CFG / guidance. |
| `num_inference_steps` | int           | `28`    | Stage 2 denoise steps. |
| `seed`                | int           | `0`     | RNG seed. |
| `pulid_weight`        | float         | `1.0`   | Identity strength (only used on a reference call). |
| `num_start_step`      | int           | `0`     | Step at which PuLID conditioning starts. |
| `true_cfg`            | float         | `1.0`   | `>1.0` enables true CFG with a negative pass. |
| `refine_denoise`      | float         | `0.20`  | Refine SDEdit strength (fraction of the trajectory re-noised). `0` disables refine. |
| `refine_steps`        | int           | `16`    | Refine schedule length. |
| `refine_guidance`     | float         | `3.0`   | Refine guidance. |
| `refine_pulid_weight` | float         | `0.0`   | Identity strength during refine (`0` = off). |
| `refine_tile_size`    | int           | `1024`  | Refine tile size (px, rounded to /16). |
| `refine_tile_overlap` | int           | `96`    | Refine tile overlap (px, feathered). |
| `upscale`             | int           | `2`     | ESRGAN upscale factor before refine. `0` skips upscaling. |

### Response

```json
{ "status": "ok", "image": "<base64 PNG>" }
```

---

## Checkpoint selection

Stage 2 runs on one of two interchangeable base transformers, chosen **per
request**:

- **`srpo`** — `FLUX.1-dev-SRPO` fine-tune. Realistic skin and detail. This is
  the default for plain text-to-image.
- **`dev`** — vanilla `FLUX.1-dev`. This is the only checkpoint PuLID identity
  conditioning was trained against, so it is used whenever a reference face is
  supplied.

### Reference vs no-reference

- **Reference call** — `use_reference: true` + `image_b64` + `file_hash`.
  Stage 1 encodes the face into `id_embeddings`; Stage 2 loads PuLID and locks
  the generated person's identity to that face. This is the face-swap path.
- **No-reference call** — `use_reference: false`, no face image. Face
  processing and PuLID are skipped; Stage 2 is plain text-to-image.

### `use_srpo` / `use_refine_step`

Both are three-state (`true` / `false` / omitted):

- **omitted (`null`)** — automatic default behaviour (see chart).
- **`true` or `false`** — forces that choice and always overrides the default.

`use_srpo: true` on a reference call loads **SRPO *and* applies PuLID**. Identity
fidelity is weaker in that combination (PuLID was not trained against SRPO), but
the explicit flag is honoured.

### Decision chart

| Reference image? | `use_srpo` | `use_refine_step` | Base checkpoint        | Refine |
|------------------|------------|-------------------|------------------------|--------|
| no               | *(omitted)* | *(omitted)*      | **SRPO**               | on     |
| yes              | *(omitted)* | *(omitted)*      | **flux1-dev** + PuLID  | on     |
| yes              | `true`     | —                 | **SRPO** + PuLID       | on     |
| no               | `false`    | —                 | **flux1-dev**          | on     |
| any              | —          | `false`           | *(as above)*           | **off** |
| any              | —          | `true`            | *(as above)*           | on     |

Resolution logic (in `run_server.generate()`):

```python
pulid_used = req.use_reference
use_srpo   = (not pulid_used) if req.use_srpo is None else bool(req.use_srpo)
use_refine = True             if req.use_refine_step is None else bool(req.use_refine_step)
```

### Swap cost

Switching between `dev` and `srpo` reloads the transformer weights in place
(`Stage2Processor._ensure_variant`) — roughly one 23 GB read from disk per
switch. The PuLID cross-attention submodules (`pulid_ca.*`) belong to neither
checkpoint and survive the swap.

Batch requests by kind — all reference calls together, then all no-reference
calls — to avoid reloading the checkpoint on every request.

---

## The refine step

Stage 3 always upscales with RealESRGAN (`upscale`, default ×2) and resizes to
the target resolution. It then optionally runs a **tiled SDEdit refine pass**
back through the Stage 2 FLUX model + VAE for extra sharpness.

Refine runs when **all** of these hold:

- `use_refine_step` is not `false` (default is on), and
- `refine_denoise > 0` (forced to `0.20` when `use_refine_step=true` and the
  caller left it at 0), and
- the Stage 2 model + VAE are available.

Refine uses whichever checkpoint Stage 2 just ran with. Set
`refine_pulid_weight > 0` to carry identity conditioning into the refine pass on
a reference call.
