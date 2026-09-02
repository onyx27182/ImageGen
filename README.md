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
| `POST` | `/generate`    | Image generation — Stage 1 (text/face encode) → Stage 2 (FLUX denoise) → Stage 3 (upscale → tiled refine → face-detail pass) |
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
| `height`              | int           | `1024`  | Final output height. FLUX is trained at 1024; smaller is draft quality. `1080x1920` is generated at half and upscaled. |
| `width`               | int           | `1024`  | Final output width. |
| `guidance_scale`      | float         | `4.0`   | Stage 2 guidance ("fake CFG"). |
| `num_inference_steps` | int           | `28`    | Stage 2 denoise steps. |
| `seed`                | int           | `0`     | RNG seed. |
| `pulid_weight`        | float         | `1.0`   | Identity strength (reference calls only). |
| `num_start_step`      | int           | `4`     | Step at which PuLID conditioning starts. PuLID docs: `~4` for photoreal, `0-1` for stylised. |
| `true_cfg`            | float         | `1.0`   | **Leave at `1.0`.** True CFG needs negative conditioning, which Stage 2 does not build — values `>1.0` will fail. |
| `refine_denoise`      | float         | `0.25`  | Tiled-refine SDEdit strength (fraction of the trajectory re-noised). `0` disables refine. |
| `refine_steps`        | int           | `24`    | Refine schedule length. Euler steps run ≈ `round(refine_steps × refine_denoise)`. |
| `refine_guidance`     | float         | `2.5`   | Refine guidance. |
| `refine_pulid_weight` | float         | `0.0`   | Identity strength during the tiled refine (`0` = off). |
| `refine_tile_size`    | int           | `1024`  | Refine tile size (px, rounded to /16). Keep ≥ 1024 for FLUX. |
| `refine_tile_overlap` | int           | `96`    | Refine tile overlap (px, feathered). |
| `upscale`             | int           | `1`     | ESRGAN factor applied before refine. `1` = none (refine at native res), `2` = supersample, `0` = also none. |
| `face_detail`         | bool          | `true`  | Run the face-detail pass (detect → crop → SDEdit at ~1024 → paste back). Fixes small features (eyes, teeth). |
| `face_denoise`        | float         | `0.40`  | Face-pass SDEdit strength. `0` disables the pass. |
| `face_pulid_weight`   | float \| null | `null`  | Identity strength in the face pass. `null` = `0.5` on reference calls, `0` otherwise. |
| `face_pad`            | float         | `0.40`  | Fraction of the face bbox added as padding before the crop. |
| `face_steps`          | int           | `30`    | Face-pass schedule length. |
| `face_guidance`       | float         | `3.0`   | Face-pass guidance. |

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
face_pw    = (0.5 if pulid_used else 0.0) if req.face_pulid_weight is None \
             else float(req.face_pulid_weight)
```

The face-detail pass (`face_detail`, default on) is independent of all of the
above — it runs on the final image regardless of checkpoint or refine choice.

### Swap cost

Switching between `dev` and `srpo` reloads the transformer weights in place
(`Stage2Processor._ensure_variant`) — roughly one 23 GB read from disk per
switch. The PuLID cross-attention submodules (`pulid_ca.*`) belong to neither
checkpoint and survive the swap.

Batch requests by kind — all reference calls together, then all no-reference
calls — to avoid reloading the checkpoint on every request.

---

## Stage 3 — upscale, refine, face pass

Stage 3 runs three steps **in this order**, all before the final resize to the
requested output size:

1. **ESRGAN upscale** — only if `upscale` is 2–4. `upscale=1` (default) skips it
   and everything below runs at the Stage 2 native resolution.
2. **Tiled SDEdit refine** — a low-denoise pass back through the Stage 2 FLUX
   model + VAE, tiled (`refine_tile_size` / `refine_tile_overlap`, feathered
   blend). Adds coherent micro-detail (hair, fabric, pores). Runs on the
   *upscaled* image, before any downscale, so features span enough pixels for
   the tiler to actually engage.
3. **Face-detail pass** (ADetailer-style) — detect the face, crop it with
   `face_pad` padding, resize the crop to ~1024 px, SDEdit it at `face_denoise`
   (with PuLID re-applied at `face_pulid_weight` on reference calls), resize
   back, paste through a feathered mask. This is the **last** diffusion step, so
   nothing re-warps the face afterwards.

Why the face pass exists: a whole-image refine cannot reliably rebuild a
~40 px iris — the model needs the feature at roughly native resolution. Step 3
gives every face ~1024 px to work with, which is what fixes warped eyes / teeth.
Order 2→3 also matters: refine before face pass, so the tiled refine never
touches the corrected face.

**Refine (step 2)** runs when `use_refine_step` is not `false`, `refine_denoise
> 0` (bumped to `0.25` if forced on at 0), and the Stage 2 model/VAE are passed
in. It uses whichever checkpoint Stage 2 just ran with.

**Face pass (step 3)** runs when `face_detail` is `true`, `face_denoise > 0`, a
face is detected, and the model/VAE are available. Detector: InsightFace
`antelopev2` (detection only), lazy-loaded, CPU.

### Best-practice reference pipeline

Matches current community practice (ADetailer, Ultimate SD Upscale, PuLID docs):

```
base gen        1024 native, dev+PuLID, guidance 4, num_start_step 4, true_cfg 1
tiled refine    @ working res, 1024 tiles, denoise 0.2–0.35   (SRPO ok here)
face pass       crop → ~1024 → denoise ~0.4, PuLID ~0.5       ← fixes eyes
output          = the resolution refined at; never up-then-downscale to a
                  smaller size expecting the refine detail to survive
```

For a bigger deliverable, either request a larger `height`/`width` directly or
set `upscale=2` (supersample: ESRGAN ×2 → refine at 2× → downscale to target).
For hero shots the field uses a dedicated upscaler (SUPIR / Gigapixel) instead
of ESRGAN — not wired in here.

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
- **omit / `true`** → run the Stage 3 tiled refine. Adds coherent micro-detail.
- **`false`** → skip refine (ESRGAN + face pass only). Faster.

**`face_detail`** — `true` / `false` &nbsp;·&nbsp; default: `true`
- **`true`** → run the face-detail pass. Keeps eyes/teeth/nostrils correct;
  costs one ~1024 px SDEdit per face (~5–10 s).
- **`false`** → skip it. Only do this for non-portrait images, or when the base
  face is already clean and you want maximum speed.

**`upscale`** — integer, use `0`/`1` or `2`–`4` &nbsp;·&nbsp; default: `1`
- **`0` / `1`** → no ESRGAN; refine and face pass run at the Stage 2 native
  resolution, output is that resolution. This is the default and is correct for
  a 1024 request.
- **`2`** → supersample: ESRGAN ×2 → refine at 2× → downscale to `height`/`width`.
  Crisper, ~2× the Stage 3 cost.
- **`3`–`4`** → bigger still; `4` is the ESRGAN model's native factor.
- **`>4`** → unsupported by the model; just a blurry resize. Don't.
- To get a genuinely larger *output*, raise `height`/`width` instead of relying
  on `upscale` — the result is resized to `height`/`width` at the very end.

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

**`height` / `width`** — integer, multiples of 16 &nbsp;·&nbsp; default: `1024`
- **`768`** → draft quality; FLUX is undertrained below 1024, anatomy is worse.
- **`1024`** → default; the resolution FLUX was trained at.
- **`1152`–`1536`** → sharper/bigger, slower, more VRAM, some OOM risk.
- **`1080×1920`** (portrait) → special-cased: generated at `540×960`, then
  upscaled to reach the target.
- This is the **final output size** — Stage 3 fits the result to it last.
- Non-multiples of 16 are rounded down.

### Identity / PuLID — reference calls only

**`pulid_weight`** — float, effective `0.0`–`1.5` &nbsp;·&nbsp; default: `1.0`
- **`0.0`** → identity off (use `use_reference: false` instead).
- **`0.5`–`0.7`** → subtle resemblance, full prompt freedom.
- **`0.8`–`1.0`** → strong likeness, still flexible (default `1.0`).
- **`1.1`–`1.3`** → maximal likeness; can fight the prompt, waxy skin, "pasted"
  look.
- **`> 1.3`** → usually degrades the image.

**`num_start_step`** — integer `0`–`num_inference_steps` &nbsp;·&nbsp; default: `4`
- **`0`–`1`** → identity from the first step: tightest likeness, but pose and
  lighting are pulled toward the reference photo. PuLID docs recommend this for
  *stylised* output.
- **`4`** → default. PuLID docs' recommendation for **photoreal**: composition
  and lighting form first, identity then locks in — more natural, likeness still
  strong.
- **`6`–`8`** → noticeably weaker likeness.
- **`> 8`** → identity barely takes effect.

**`true_cfg`** — **keep at `1.0`** &nbsp;·&nbsp; default: `1.0`
- Negative conditioning is not built by Stage 2, so any value `> 1.0` errors.

### Tiled refine — Stage 3, step 2

Actual Euler steps executed ≈ `round(refine_steps × refine_denoise)`.

**`refine_denoise`** — float `0.0`–`1.0` &nbsp;·&nbsp; default: `0.25`
- **`0.0`** → refine disabled.
- **`0.15`–`0.25`** → gentle polish; output stays faithful to Stage 2 (default `0.25`).
- **`0.3`–`0.4`** → real re-detailing (skin, hair, fabric); mild drift in
  identity/composition. `> 0.5` on a single non-tiled pass is what warps faces —
  keep it in this band and let the face pass handle the face.
- **`0.5`–`0.6`** → strong reinterpretation; identity and layout move.
- **`> 0.7`** → effectively regenerates each tile from the prompt.
- Community consensus for tiled upscale refine is `0.2–0.35`; `>0.5` drifts into
  patchwork.

**`refine_steps`** — integer &nbsp;·&nbsp; default: `24`
- **`12`–`16`** → faster, slightly coarser.
- **`24`** → default.
- **`32`–`40`** → smoother, marginal gain (raises the *actual* step count, which
  is `round(refine_steps × refine_denoise)`).

**`refine_guidance`** — float &nbsp;·&nbsp; default: `2.5`
- **`1.5`–`2.5`** → refine follows the existing image (default `2.5`).
- **`3.0`–`3.5`** → more prompt influence.
- **`> 4`** → prompt overrides image content; contrast rises.

**`refine_pulid_weight`** — float `0.0`–`1.0` &nbsp;·&nbsp; default: `0.0`
- **`0.0`** → the tiled refine ignores identity (fine at `refine_denoise` ≤ 0.3;
  the face pass re-asserts identity anyway).
- **`0.3`–`0.6`** → hold identity through a heavier refine.
- Reference calls only (needs `pulid_ca` in the model).

**`refine_tile_size`** — integer px, rounded to ÷16 &nbsp;·&nbsp; default: `1024`
- **`< 1024`** → **avoid on FLUX** — tiles below native resolution produce
  repeated / incoherent content.
- **`1024`** → default; FLUX native.
- **`1152`–`1280`** → fewer tiles, more global coherence, more VRAM per tile.
- **≥ the image's larger side** → single tile, no seams.

**`refine_tile_overlap`** — integer px, capped at `refine_tile_size / 2` &nbsp;·&nbsp; default: `96`
- **`0`–`48`** → fastest, risk of visible grid seams.
- **`64`–`128`** → smooth blends (default `96`).
- **`> 128`** → more tiles ⇒ slower, little visible gain.

### Face-detail pass — Stage 3, step 3

**`face_denoise`** — float `0.0`–`0.7` &nbsp;·&nbsp; default: `0.40`
- **`0.0`** → face pass disabled (same as `face_detail: false`).
- **`0.25`–`0.35`** → light cleanup; keeps every base-face detail, fixes only
  minor issues.
- **`0.4`** → default. Rebuilds eyes/teeth/nostrils cleanly; the face is now at
  ~1024 px so higher denoise is safe here (unlike the whole-image refine).
  Matches ADetailer's default.
- **`0.5`–`0.6`** → stronger; can pull the face slightly off the reference —
  raise `face_pulid_weight` to compensate.
- **`> 0.6`** → the face starts to become a different face.

**`face_pulid_weight`** — float `0.0`–`1.0`, or `null` &nbsp;·&nbsp; default: `null`
- **`null`** → auto: `0.5` on reference calls, `0.0` otherwise.
- **`0.4`–`0.6`** → keeps the reference identity through the face pass.
- **`0.0`** → face pass follows the prompt only (correct for non-reference).
- **`> 0.7`** → can re-introduce the "pasted-on" look.

**`face_pad`** — float &nbsp;·&nbsp; default: `0.40`
- Fraction of the detected face box added as margin before cropping.
- **`0.2`–`0.3`** → tight crop, more resolution on the features, but the paste
  boundary sits closer to the face (seam risk on strong lighting).
- **`0.4`** → default; includes brow/jaw/some hair for a safe blend.
- **`0.6`–`0.8`** → includes neck/ears; less feature resolution.

**`face_steps`** — integer &nbsp;·&nbsp; default: `30`  ·  **`face_guidance`** — float &nbsp;·&nbsp; default: `3.0`
- Same meaning as the refine equivalents, scoped to the face crop. Defaults are
  fine; raise `face_guidance` to `3.5` only if the face ignores prompt cues.

### Rules of thumb

- **Warped / mismatched eyes** → make sure `face_detail: true` (default) and
  `height` ≥ 1024. Raise `face_denoise` to `0.45`.
- **Face pass changed the likeness** → raise `face_pulid_weight` to `0.6`, or
  lower `face_denoise` to `0.3`.
- **Visible edge around the face after the pass** → raise `face_pad` to `0.55`.
- **Face looks pasted on / wrong lighting** → lower `pulid_weight` to ~0.8, or
  raise `num_start_step` to `4`–`6`.
- **Likeness lost after the tiled refine** → lower `refine_denoise`, or set
  `refine_pulid_weight` to ~0.5.
- **Output too soft** → raise `refine_denoise` to ~0.3, raise
  `num_inference_steps`, or set `upscale: 2`.
- **Skin looks plasticky** → `use_srpo: true` (drop the reference image, or
  accept weaker likeness).
- **Visible grid seams after refine** → raise `refine_tile_overlap`, or raise
  `refine_tile_size` so the image fits in one tile.
- **Too slow** → `face_detail: false` and/or `use_refine_step: false`, drop
  `num_inference_steps` to ~22, keep `height`/`width` at 1024 and `upscale` at 1.

## Other endpoints

`POST /change_view` — body `{ "image_b64": ..., "prompts": [...] }`, no tuning
knobs; streams one NDJSON line per prompt.

`POST /dolphin` — body `{ "prompt": ..., "max_new_tokens": 512 }`;
`max_new_tokens` is the only knob.
