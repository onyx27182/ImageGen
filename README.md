# ImageGen

Multi-model FastAPI image-generation server. One GPU (A100 80GB); only one
pipeline is resident on the GPU at a time and models are hot-swapped between
requests.

## Installation

First-time setup — Python deps, the vendored PuLID-FLUX tree, and ~150 GB of
model weights — is documented step by step in **[INSTALL.md](INSTALL.md)**.

Quick version, on a host that already meets the [prerequisites](INSTALL.md#1-host-prerequisites):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

- `requirements.txt` — curated, pinned deps the server imports.
- `requirements.lock.txt` — full `pip freeze` for a byte-exact environment clone.

The `--extra-index-url` is required: the pinned `torch`/`torchvision` are CUDA
13.0 builds that are not on PyPI. `pip` alone will not resolve them.

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
| `POST` | `/generate`    | Image generation. Default = SRPO Stage 2 only. Opt-in stages: PuLID (`use_reference`), ESRGAN `upscale`, tiled `use_refine_step`, `face_detail` pass. |
| `POST` | `/change_view` | Camera-angle change via QwenImageEdit angles LoRA (NDJSON stream) |
| `POST` | `/dolphin`     | LLM chat (vLLM, lazy-loaded on first call) |

All endpoints require the `X-API-KEY` header.

### Long renders and the proxy timeout

A full render (Stage 2 + `upscale` + refine + face pass, when enabled) can run for
several minutes.
Cloudflare, which fronts the `*.thundercompute.net` subdomain, returns
**`524`** if the origin sends nothing for ~100 s — so a plain blocking `POST`
that renders inline gets cut off before the image is done.

`/generate` handles this transparently, with **no client changes** (it relies
only on `requests` following redirects, which it does by default):

1. `POST /generate` validates the request, starts the render on a background
   thread, and immediately returns **`303 See Other`** → `/generate/wait/{id}`.
2. `requests` follows that to `GET /generate/wait/{id}`, which waits up to
   `WAIT_SECONDS` (45 s) for the render.
3. If the render finished, the wait endpoint returns the **final response** —
   `200 { "status": "ok", "image": "..." }`, or the real `4xx`/`5xx` with
   `{ "detail": "..." }` if it failed.
4. If it is still running, the wait endpoint `303`s back to itself and the
   client comes round again.

Every HTTP hop is well under the proxy timeout, so `524` never happens. From the
client's side it is still a single blocking `POST` that returns the true status
code and body — a failed render is a real error response, at any duration.
After `MAX_HOPS` (20 ≈ 15 min) the wait endpoint returns `504`.

Tunables in `run_server.py`: `WAIT_SECONDS`, `MAX_HOPS`, `RENDER_TTL`.

---

## `POST /generate`

### Request body

**A bare request — just `prompt` — runs SRPO Stage 2 and nothing else.** Every
other stage is opt-in:

| Toggle | Type | Default | Turns on |
|---|---|---|---|
| `use_srpo`        | bool | **`true`** | SRPO checkpoint. Runs **always** unless you send `false` (→ vanilla flux1-dev). |
| `use_reference`   | bool | `false` | PuLID face conditioning. Needs `image_b64` + `file_hash`. |
| `use_refine_step` | bool | `false` | Stage 3 SDEdit refine pass. |
| `face_detail`     | bool | `false` | Stage 3 face-detail pass. See [when to use it](#when-to-use-the-face-pass). |
| `upscale`         | int `0`–`4` | `1` | ESRGAN supersample + tiled refine (`2`+). |

Everything else is a tuning parameter:

| Field                 | Type          | Default | Notes |
|-----------------------|---------------|---------|-------|
| `prompt`              | str           | *(required)* | Text prompt |
| `image_b64`           | str \| null   | `null`  | Base64 PNG/JPEG of the reference face. Required when `use_reference=true`. |
| `file_hash`           | str           | `""`    | Stable key for the reference image; used to cache the face embedding. Required when `use_reference=true`. |
| `height`              | int           | `1024`  | Base output height. FLUX is trained at 1024; smaller is draft quality. `1080x1920` is generated at half and upscaled. With `upscale>=2` the returned image is `height × upscale`. |
| `width`               | int           | `1024`  | Base output width (see `height`). |
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
| `upscale`             | int (`0`–`4`) | `1`     | ESRGAN factor before refine. `1`/`0` = none. `2`–`4` = supersample + **tiled** refine, image returned at `gen_size × upscale` (no downscale). `4` from 1024 = 4096² / ~25 tiles / minutes. `>4` → `422`. |
| `face_denoise`        | float         | `0.40`  | Face-pass SDEdit strength. `0` disables the pass. |
| `face_pulid_weight`   | float \| null | `null`  | Identity strength in the face pass. `null` = `0.5` on reference calls, `0` otherwise. |
| `face_pad`            | float         | `0.40`  | Fraction of the face bbox added as padding before the crop. |
| `face_steps`          | int           | `30`    | Face-pass schedule length. |
| `face_guidance`       | float         | `3.0`   | Face-pass guidance. |

### Response

```json
{ "status": "ok", "image": "<base64 PNG>" }
```

One blocking call. Internally it is delivered via a `303` redirect chain so long
renders survive the proxy timeout — see
[Long renders and the proxy timeout](#long-renders-and-the-proxy-timeout). The
client only needs to follow redirects (the default) and use a read timeout
longer than `WAIT_SECONDS` (or none).

---

## Checkpoint selection

Stage 2 runs on one of two interchangeable base transformers, chosen **per
request**:

- **`srpo`** — `FLUX.1-dev-SRPO` fine-tune. Realistic skin and detail. **This is
  the default for every call**, reference or not.
- **`dev`** — vanilla `FLUX.1-dev`. Only used when `use_srpo` is explicitly sent
  as `false`. PuLID identity conditioning was trained against `dev`, so a
  reference call gets slightly stronger likeness on `dev` — send `use_srpo:
  false` if likeness matters more than SRPO's skin/detail rendering.

### Reference vs no-reference

- **Reference call** — `use_reference: true` + `image_b64` + `file_hash`.
  Stage 1 encodes the face into `id_embeddings`; Stage 2 loads PuLID and locks
  the generated person's identity to that face. This is the face-swap path.
- **No-reference call** — `use_reference: false`, no face image. Face
  processing and PuLID are skipped; Stage 2 is plain text-to-image.

### The four toggles

Plain booleans. A bare request runs **SRPO Stage 2 only**.

| `use_srpo` | `use_reference` | `use_refine_step` | `face_detail` | Result |
|---|---|---|---|---|
| *(omit)* | *(omit)* | *(omit)* | *(omit)* | **SRPO, Stage 2 only** |
| `false` | — | — | — | vanilla flux1-dev, Stage 2 only |
| — | `true` | — | — | SRPO **+ PuLID** (needs `image_b64`+`file_hash`) |
| `false` | `true` | — | — | flux1-dev + PuLID *(strongest likeness — PuLID was trained on dev)* |
| — | — | `true` | — | SRPO + Stage 3 refine |
| — | — | — | `true` | SRPO + face-detail pass |

Combine freely. `upscale >= 2` additionally turns on the ESRGAN supersample and
makes the refine genuinely tiled.

Resolution logic (`run_server._run_generate()`):

```python
pulid_used = bool(req.use_reference)
use_srpo   = bool(req.use_srpo)          # default True
use_refine = bool(req.use_refine_step)   # default False
face_pw    = (0.5 if pulid_used else 0.0) if req.face_pulid_weight is None \
             else float(req.face_pulid_weight)
# req.face_detail (default False) flows straight through to Stage 3
```

### Swap cost

Switching between `dev` and `srpo` reloads the transformer weights in place
(`Stage2Processor._ensure_variant`) — roughly one 23 GB read from disk per
switch. The PuLID cross-attention submodules (`pulid_ca.*`) belong to neither
checkpoint and survive the swap.

Batch requests by kind — all reference calls together, then all no-reference
calls — to avoid reloading the checkpoint on every request.

---

## Stage 3 — upscale, refine, face pass

Stage 3 has three steps, **each independently gated** — a default request skips
all of them. When enabled, they run in this order:

1. **ESRGAN upscale** — only if `upscale` is 2–4. `upscale=1` (default) skips it
   and everything below runs at the Stage 2 native resolution.
2. **SDEdit refine** — only if `use_refine_step: true`. A low-denoise pass back
   through the Stage 2 FLUX model + VAE. **It only *tiles* when the image is
   larger than one tile**
   (`refine_tile_size`, default 1024) — i.e. only after an `upscale >= 2`. At
   `upscale=1` the image is 1024 and the refine is a single full-frame pass, not
   a tiled one. Adds coherent micro-detail (hair, fabric, pores).
3. **Face-detail pass** (ADetailer-style) — only if `face_detail: true`. Detect
   the face, crop it (+`face_pad`), scale the crop to ~1024 px, SDEdit it at
   `face_denoise`, scale back, paste through a feathered mask. Overall image size
   is unchanged. This is the **last** diffusion step, so nothing re-warps the
   face afterwards. Full mechanism: [What the face pass does](#what-the-face-pass-does).

**The output is returned at the resolution steps 2–3 ran at.** With `upscale >= 2`
you get the full supersampled image back (e.g. `1024` request + `upscale=2` → a
`2048` PNG); it is **not** downscaled to `height`/`width`, because that would
throw the refine detail straight back out.

Why the face pass exists: a whole-image refine cannot reliably rebuild a
~40 px iris — the model needs the feature at roughly native resolution. Step 3
gives every face ~1024 px to work with, which is what fixes warped eyes / teeth.
Order 2→3 also matters: refine before face pass, so the tiled refine never
touches the corrected face.

**Refine (step 2)** runs only when `use_refine_step: true` is sent (`refine_denoise`
then bumped to `0.25` if it was left at 0). It uses whichever checkpoint Stage 2
ran with.

**Face pass (step 3)** runs only when `face_detail: true` is sent, `face_denoise
> 0`, and a face is detected. Detector: InsightFace `antelopev2` (detection only),
lazy-loaded, CPU.

#### What the face pass does

Per detected face (`Stage3Processor._face_pass`), on the image as it stands after
step 2:

1. **Detect** — InsightFace `antelopev2` (detection only, CPU, `det_size 640`)
   returns face bounding boxes. No face → the image is returned unchanged. Up to
   `face_max` (2) faces, largest first.
2. **Crop** — take the bbox, expand it by `face_pad` (0.40 ⇒ +40 % on each side),
   clamp to the image, round the crop to ÷16. This crop is at the image's
   **native pixels** — call it `cw × ch`.
3. **Scale the crop to the work size** — `scale = face_work_size / max(cw, ch)`
   (`face_work_size` default 1024), **LANCZOS**. So the crop's long side becomes
   ~1024 px. If the face was **smaller** than that, this is an **upscale** (and
   the point of the whole pass). If the face was **larger** (a tight portrait),
   this is a **downscale**.
4. **Re-diffuse the scaled crop** — the same partial SDEdit as the refine
   (`_refine_tile`): VAE-encode → re-noise to `face_denoise` (0.40) → run the
   `face_steps` (30) schedule ≈ 12 actual denoise steps through the FLUX
   transformer, conditioned on the text prompt, plus PuLID identity if
   `face_pulid_weight > 0`.
5. **Scale back** to `cw × ch`, LANCZOS.
6. **Paste** into the image through a Gaussian-feathered rectangular mask
   (feather ≈ `min(cw, ch) / 8`).

The **overall image dimensions never change** — only the face region is
replaced, in place. Consequence: on a large face the crop makes a
downscale → re-diffuse → upscale round-trip that **loses** detail; it only adds
detail when step 3 is a real upscale.

#### When to use the face pass

It only helps when the detected face is **small relative to the work resolution**,
so that cropping it and scaling to ~1024 px is a real upscale that gives the model
pixels to rebuild irises / teeth / nostrils.

| Situation | `face_detail` | Why |
|---|---|---|
| Full-body, wide, or group shot — face is a small part of the frame | **`true`** | The crop→1024 step is a genuine upscale; fixes warped small features. |
| After `upscale >= 2` | **`true`** | Same — the face crop gets a proper high-res redo. |
| Tight portrait at native 1024, no upscale | **`false`** | The face already fills the frame, so the crop is already ~1024 — no resolution gained. You get a marginal eye sharpen plus a **paste-seam risk**: the feathered crop boundary can show as a faint ring on flat skin (foreheads), and it slightly smooths skin texture. |
| You are chasing skin texture | **`false`** | The pass re-diffuses the face and pulls it toward FLUX's smooth-skin prior. |

If you do run it on a close portrait, drop `face_pad` to ~`0.2` so the crop hugs
the face and the blend edge stays off open skin.

### Recipes

**Cleanest skin (default):** just `prompt`. SRPO Stage 2, 1024, nothing else.
Add `guidance_scale: 3.5`, `num_inference_steps: 50` (SRPO's own recommended
settings) and skin-texture words in the prompt for the best result.

**Face swap:** `use_reference: true` + `image_b64` + `file_hash`. Optionally
`use_srpo: false` for the strongest likeness (PuLID was trained on dev).

**Crisp large deliverable:** `upscale: 2` + `use_refine_step: true` → ESRGAN ×2,
genuinely tiled 1024 refine at 2048, `2048` PNG returned (no downscale).

**Small / distant face with warped eyes:** `face_detail: true` (see
[when to use it](#when-to-use-the-face-pass)).

For hero shots the field uses a dedicated upscaler (SUPIR / Gigapixel) instead of
ESRGAN — not wired in here.

---

## Quality & tuning knobs

Every knob below is a field on the `POST /generate` body. Each entry gives the
**accepted range**, the **default**, and what specific values do. Defaults are
tuned for a good general result; adjust one knob at a time.

### Pipeline selection

**`use_srpo`** — bool &nbsp;·&nbsp; default: **`true`**
- **`true` / omit** → SRPO, on **every** call. Realistic skin, pores,
  micro-contrast. On a reference call PuLID runs on top of it.
- **`false`** → vanilla flux1-dev. Gives PuLID its strongest likeness (PuLID was
  trained on `dev`); look is cleaner and slightly flatter.

**`use_reference`** — bool &nbsp;·&nbsp; default: `false`
- **`true`** → PuLID face conditioning; requires `image_b64` + `file_hash`.
- **`false` / omit** → plain text-to-image.

**`use_refine_step`** — bool &nbsp;·&nbsp; default: `false`
- **`true`** → run the Stage 3 refine pass. It only *tiles* at `upscale >= 2`; at
  `upscale=1` it's a single full-frame pass that tends to **smooth skin** (it
  re-diffuses toward FLUX's smooth prior).
- **`false` / omit** → skip it.

**`face_detail`** — bool &nbsp;·&nbsp; default: `false`
- **`true`** → run the face-detail pass (one ~1024 px SDEdit per face, ~5–10 s).
  Worth it only when the face is **small in frame** or after `upscale >= 2`.
- **`false` / omit** → skip it. Correct for a **tight portrait at native 1024** —
  the face is already ~1024 px so the pass adds nothing but a paste-seam risk and
  slight skin smoothing.
- Full guidance: [When to use the face pass](#when-to-use-the-face-pass).

**`upscale`** — integer, use `0`/`1` or `2`–`4` &nbsp;·&nbsp; default: `1`
- **`0` / `1`** → no ESRGAN; refine is a **single full-frame pass** (not tiled)
  and the output is the Stage 2 native resolution. Fine for a plain 1024 request,
  but the refine is doing less here than the word "tiled" suggests.
- **`2`** → ESRGAN ×2, then a genuinely **tiled** 1024 refine at 2×, then the
  face pass — and the **`2048` image is returned as-is**, no downscale. Crisper,
  ~2× the Stage 3 cost. Use this when you want the full refined result.
- **`3`** → ESRGAN ×3, refine tiled at 3072 (~16 tiles), output `gen_size × 3`.
- **`4`** → the ESRGAN model's native factor. From a 1024 request: a **4096×4096**
  output, **~25 refine tiles**, several minutes, and a PNG in the tens of MB
  (base64 in the JSON body). More tiles also means more seam-blend surface, so
  it is not automatically "better" than `2` — use it only when you actually need
  a 4K deliverable.
- **`>4`** → rejected (`422`). It would only be a blurry `cv2` resize on top of
  the ×4 network.

For most work `2` is the sweet spot: real tiled refine, 2048 out, manageable
cost and payload.

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
  ESRGAN-upscaled ×2 to reach the target.
- This is the **base output size**. With `upscale >= 2` the returned image is
  `height × upscale` / `width × upscale` — Stage 3 does **not** shrink it back.
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

- **Warped / mismatched eyes** on a small or distant face → `face_detail: true`
  (off by default), `height` ≥ 1024, `face_denoise` ~`0.45`. If the face is
  already large in frame this won't help — see
  [when to use the face pass](#when-to-use-the-face-pass).
- **Visible ring around the face after the pass** → drop `face_pad` to ~`0.2`, or
  turn `face_detail` back off.
- **Face looks pasted on / wrong lighting** (reference call) → lower
  `pulid_weight` to ~0.8, or raise `num_start_step` to `4`–`6`.
- **Skin looks plasticky / waxy** → the default (SRPO Stage 2 only) already gives
  the most texture. Do **not** turn on `use_refine_step` or `face_detail` — both
  re-diffuse toward FLUX's smooth prior. Lower `guidance_scale` to ~3.0, raise
  `num_inference_steps` to 50, and put skin-texture words in the prompt. Residual
  highlight waxiness is FLUX-family baseline.
- **Output too soft / low-res** → `upscale: 2` + `use_refine_step: true` → the
  full tiled-refined `2048` PNG.
- **Visible grid seams after refine** → raise `refine_tile_overlap`, or raise
  `refine_tile_size` so the image fits in one tile.
- **Too slow** → the default (SRPO Stage 2 only) is already the fast path; drop
  `num_inference_steps` to ~22 and don't enable `upscale` / `use_refine_step` /
  `face_detail`.

## Other endpoints

`POST /change_view` — body `{ "image_b64": ..., "prompts": [...] }`, no tuning
knobs; streams one NDJSON line per prompt.

`POST /dolphin` — body `{ "prompt": ..., "max_new_tokens": 512 }`;
`max_new_tokens` is the only knob.
