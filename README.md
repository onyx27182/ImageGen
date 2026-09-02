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
| `true_cfg`            | float         | `1.0`   | **Leave at `1.0`.** True CFG needs negative conditioning, which Stage 2 does not currently build — values `>1.0` will fail. |
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

---

## Quality & tuning knobs

Every knob below is a field on the `POST /generate` body. Each entry gives the
**accepted range**, the **default**, and what specific values do. Defaults are
tuned for a good general result; adjust one knob at a time.

### Pipeline selection

**`use_srpo`** — `true` / `false` / omit &nbsp;·&nbsp; default: omit (auto)
- **omit** → auto: SRPO when there is no reference image, flux1-dev when there is.
- **`true`** → always SRPO. Realistic skin, pores, micro-contrast. Use for
  non-face images or when a loose likeness is acceptable.
- **`false`** → always flux1-dev. Required for a strong PuLID likeness; look is
  cleaner and slightly flatter.

**`use_refine_step`** — `true` / `false` / omit &nbsp;·&nbsp; default: omit (on)
- **omit / `true`** → run the Stage 3 diffusion refine after upscaling
  (~2× Laplacian sharpness at `refine_denoise` 0.4). Adds ~5–15 s.
- **`false`** → ESRGAN upscale only. Faster, and nothing can drift.

**`upscale`** — integer, use `0` or `2`–`4` &nbsp;·&nbsp; default: `2`
- **`0`** → no upscaling; refine (if on) runs at the Stage 2 native size.
- **`2`** → ×2 output (default). Good detail/size balance.
- **`3`–`4`** → larger output, more ESRGAN texture; `4` is the model's native
  factor. Slower, more VRAM in refine.
- **`>4`** → not supported by the ESRGAN model; extra scale is just a resize and
  looks soft. Don't.

### Base generation — Stage 2

**`num_inference_steps`** — integer &nbsp;·&nbsp; default: `28`
- **`< 15`** → under-denoised: soft, muddy, incoherent.
- **`20`–`28`** → normal working range; `28` is the sweet spot.
- **`30`–`40`** → marginally crisper, linear cost increase.
- **`> 40`** → no visible benefit.

**`guidance_scale`** — float &nbsp;·&nbsp; default: `4.0`
- **`1.5`–`2.5`** → very loose; dreamy, washed-out, ignores prompt details.
- **`3.5`–`4.5`** → balanced prompt adherence and natural look (default `4.0`).
- **`5`–`7`** → follows the prompt hard; rising contrast/saturation, "AI" sheen.
- **`> 8`** → over-saturated, burnt highlights, artifacts.

**`seed`** — non-negative integer &nbsp;·&nbsp; default: `0`
- Same seed + identical params ⇒ identical image. Change it to get a different
  sample; nothing else about quality changes.

**`height` / `width`** — integer, multiples of 16 &nbsp;·&nbsp; default: `768`
- **`512`** → fast, lower fidelity, weaker composition.
- **`768`** → default; reliable for FLUX.
- **`1024`** → sharper, better structure; ~1.8× slower, more VRAM.
- **`1152`–`1536`** → diminishing returns, slow, OOM risk.
- **`1920×1080`** → special-cased: generated at `960×540`, then upscaled.
- Non-multiples of 16 are rounded down.

### Identity / PuLID — reference calls only

**`pulid_weight`** — float, effective `0.0`–`1.5` &nbsp;·&nbsp; default: `1.0`
- **`0.0`** → identity off (use `use_reference: false` instead).
- **`0.5`–`0.7`** → subtle resemblance, full prompt freedom.
- **`0.8`–`1.0`** → strong likeness, still flexible (default `1.0`).
- **`1.1`–`1.3`** → maximal likeness; can fight the prompt, waxy skin, "pasted"
  look.
- **`> 1.3`** → usually degrades the image.

**`num_start_step`** — integer `0`–`num_inference_steps` &nbsp;·&nbsp; default: `0`
- **`0`** → identity applied from the first step: tightest likeness, but pose and
  lighting are pulled toward the reference photo.
- **`1`–`3`** → composition forms first, then identity: more natural result,
  likeness slightly looser.
- **`4`–`6`** → noticeably weaker likeness.
- **`> 6`** → identity barely takes effect.

**`true_cfg`** — **keep at `1.0`** &nbsp;·&nbsp; default: `1.0`
- Negative conditioning is not built by Stage 2, so any value `> 1.0` errors.

### Refine pass — Stage 3

Only applied when the refine step runs. Actual Euler steps executed ≈
`round(refine_steps × refine_denoise)`.

**`refine_denoise`** — float `0.0`–`1.0` &nbsp;·&nbsp; default: `0.20`
- **`0.0`** → refine disabled (ESRGAN only).
- **`0.1`–`0.2`** → gentle polish; output stays faithful to Stage 2 (default `0.20`).
- **`0.3`–`0.4`** → real re-detailing (skin, hair, fabric); mild drift in
  identity/composition. Best quality/faithfulness trade for portraits ≈ `0.35`.
- **`0.5`–`0.6`** → strong reinterpretation; expect identity and layout to move.
- **`> 0.7`** → effectively regenerates each tile from the prompt.

**`refine_steps`** — integer &nbsp;·&nbsp; default: `16`
- **`8`–`12`** → faster, slightly coarser refine.
- **`16`** → default.
- **`20`–`24`** → smoother, marginal quality gain.
- **`> 24`** → not worth the time.

**`refine_guidance`** — float &nbsp;·&nbsp; default: `3.0`
- **`1.5`–`2.5`** → refine mostly follows the existing image.
- **`3.0`–`3.5`** → balanced (default `3.0`).
- **`> 4`** → prompt can override image content during refine; contrast rises.

**`refine_pulid_weight`** — float `0.0`–`1.0` &nbsp;·&nbsp; default: `0.0`
- **`0.0`** → refine ignores identity (fine when `refine_denoise` ≤ 0.25).
- **`0.3`–`0.6`** → re-asserts the reference face; use when `refine_denoise`
  ≥ 0.35 has eroded the likeness.
- **`> 0.7`** → can re-introduce the "pasted" look.
- Only has an effect on a reference call (needs `pulid_ca` in the model).

**`refine_tile_size`** — integer px, rounded to ÷16 &nbsp;·&nbsp; default: `1024`
- **`768`** → lowest VRAM, more tiles, more seams to blend.
- **`1024`** → default balance.
- **`1152`–`1280`** → fewer tiles, more global coherence, more VRAM per tile.
- **≥ the image's larger side** → single tile, no seams at all (best if it fits).

**`refine_tile_overlap`** — integer px, capped at `refine_tile_size / 2` &nbsp;·&nbsp; default: `96`
- **`0`–`32`** → fastest, risk of visible grid seams.
- **`64`–`128`** → smooth blends (default `96`).
- **`> 128`** → smoother still but more tiles ⇒ slower, little visible gain.

### Rules of thumb

- **Face looks pasted on / wrong lighting** → lower `pulid_weight` to ~0.8, or
  raise `num_start_step` to 2–4.
- **Face likeness lost after refine** → lower `refine_denoise`, or set
  `refine_pulid_weight` to ~0.5.
- **Output too soft** → `use_refine_step: true` and `refine_denoise` ~0.3, or
  raise `num_inference_steps`.
- **Skin looks plasticky** → `use_srpo: true` (drop the reference image, or
  accept weaker likeness).
- **Visible grid seams after refine** → raise `refine_tile_overlap`, or raise
  `refine_tile_size` so the image fits in one tile.
- **Too slow** → `use_refine_step: false`, drop `num_inference_steps` to ~22,
  keep resolution at 768.

## Other endpoints

`POST /change_view` — body `{ "image_b64": ..., "prompts": [...] }`, no tuning
knobs; streams one NDJSON line per prompt.

`POST /dolphin` — body `{ "prompt": ..., "max_new_tokens": 512 }`;
`max_new_tokens` is the only knob.
