# Image generation and editing

The `image-flux3` service runs FLUX.3 Image Klein on port `:9196`. It serves the
OpenAI image routes for normal work and a native async API for hires passes,
fixed seeds and sampler control. Both routes use the same model and the same
defaults: 24 steps, `dpm++2m` with the `karras` scheduler, strength 0.9 for
edits, and `RealESRGAN_x4` for hires. A 1024x1024 render takes about 27 s, plus
about 1 s for prompt expansion; a 1248x1824 hires render (1.5x from 832x1216)
takes about 128 s plus prompt expansion.

## Quick start

`scripts/flux_client.py` is a command line client built on the Python standard
library. It needs Python 3.10 or newer and no packages.

The client always expands the prompt through the local chat model before it
renders. That step needs the `llm` profile. Pass `--no-expand` to send the prompt
as-is, and `--show-prompt` to see the caption that was sent.

```bash
./scripts/flux_client.py generate "a red fox in a snowy forest, evening light" --out fox.png
./scripts/flux_client.py edit "change the mug to bright blue" mug.png --out mug-blue.png
./scripts/flux_client.py hires "a red fox in a snowy forest, evening light" --scale 1.5 --out fox-large.png
```

The client talks to the image service at `http://localhost:9196` and the chat
service at `http://localhost:9193` by default. Point it at another host with
`--api` and `--chat-api`, or with the `SPARKSTACK_IMAGE_URL` and
`SPARKSTACK_CHAT_URL` environment variables:

```bash
./scripts/flux_client.py generate "a red fox in a snowy forest" \
  --api http://192.168.1.10:9196 --chat-api http://192.168.1.10:9193 \
  --out fox.png
```

It prints the output path, the image dimensions and the time the job took. Add
`--seed` to reproduce a result. The seed is sent as a top-level field, which is
the only place `sd-server` reads it. Individual pixels can still differ between
runs because the GPU kernels are not bit-exact, so compare images visually.

## Prompt expansion

The model expects a full scene caption. The client sends the request through the
chat model with the templates in `config/image/`, adapted from BFL's FLUX.3
prompt upsampler, then renders the returned caption. Expansion may reword or
drop details, so pass `--no-expand` with a full caption when exact wording
matters, for example character continuity across a series. Edits use the first image
as context for the expansion. Multi-reference edits still send every reference
to the image model, but the caption writer sees only the first one.

Direct API calls bypass this step. If you call the OpenAI or native route
without the client, send a full caption instead of a short request.

## Generate

The OpenAI route is the shortest path for simple integrations. It returns JSON,
not image bytes, so decode `data[0].b64_json` before writing the file. The raw
route does not expand prompts; the example uses a caption rather than a short
request:

```bash
curl -s http://192.168.1.10:9196/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Full-color outdoor photograph, a red fox stands in a snowy forest at dusk, its orange fur catching the last light, dark evergreens behind it and soft snow in the foreground.", "size": "1024x1024"}' \
  | python3 -c 'import json,sys,base64; open("fox.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The server generates at 1024x1024 when `size` is absent. That is the model's
native size and the best detail per second. A direct 2048x2048 render works, but
it is softer than a hires pass and takes several times longer.

The route accepts `<sd_cpp_extra_args>` in the prompt for per-request overrides.
Sampler fields go under `sample_params`, but `seed` is top-level:

```text
a red fox in a snowy forest <sd_cpp_extra_args>{"sample_steps":16,"seed":42}</sd_cpp_extra_args>
```

The tag is removed before the prompt reaches the model. Both routes return PNG
data by default. The native API below is easier to use when a client needs to
change several settings.

## Edit

`POST /v1/images/edits` uses multipart form data. Send one or more `image`
fields. The first image becomes the img2img init image, and every image is
passed to the model as a reference. The prompt should describe the finished
scene, not only the change. The Python client expands the instruction first,
using the first image as context.

```bash
curl -s http://192.168.1.10:9196/v1/images/edits \
  -F 'prompt=change the mug to bright blue, keep everything else the same' \
  -F image=@mug.png \
  | python3 -c 'import json,sys,base64; open("mug-blue.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`strength` controls how much of the source is repainted. The service starts at
0.9 because 0.75 often leaves large subjects mostly unchanged. For a subtler
edit, pass a lower value in the prompt:

```text
repaint the mug in a bright blue glaze <sd_cpp_extra_args>{"strength":0.6}</sd_cpp_extra_args>
```

Do not expect an instruction editor. Klein follows the caption only weakly: at
strength 0.9 it can return the same subject in a fresh render instead of changing
its color, and text or object replacement is unreliable. Use it for styling,
framing and incidental variation, and compose the change in post when identity
must change.

The native API takes `strength` as a normal JSON field. Send the source image as
a base64 string in `init_image` and any extra references in `ref_images`; the
Python client encodes both for you and does not send the init image again as a
reference.

FLUX.3 Klein is an img2img model with reference conditioning, not an
instruction editor. A reference image on its own tends to reproduce the
reference instead of editing it, and masked inpaint is unreliable. Prompt-driven
img2img is the path that works.

## Native async API

Submit a job to `POST /sdcpp/v1/img_gen`:

```bash
curl -s http://192.168.1.10:9196/sdcpp/v1/img_gen \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Full-color outdoor photograph, a red fox stands in a snowy forest at dusk, orange fur catching the last light, dark evergreens behind it.",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "sample_params": {"sample_steps": 24, "sample_method": "dpm++2m", "scheduler": "karras"}
  }'
```

The service returns a job with an id, a status and a poll URL:

```json
{"id": "job_6ad37833_00000000", "status": "queued", "poll_url": "/sdcpp/v1/jobs/job_6ad37833_00000000"}
```

Poll the URL until `status` is `completed` or `failed`:

```bash
curl -s http://192.168.1.10:9196/sdcpp/v1/jobs/job_6ad37833_00000000
```

A completed job stores the image at `result.images[0].b64_json`. The `error`
field is null on success. The client in `scripts/flux_client.py` handles the
submit and poll loop for you.

## Hires and large output

Add a `hires` object to a native job for a two-pass render. The first pass makes
the base image; the second pass upscales it and repaints at the target scale.

```json
{
  "prompt": "Full-color outdoor photograph, a red fox stands in a snowy forest at dusk, orange fur catching the last light, dark evergreens behind it.",
  "width": 1024,
  "height": 1024,
  "hires": {
    "enabled": true,
    "upscaler": "RealESRGAN_x4",
    "scale": 1.5,
    "denoising_strength": 0.3,
    "steps": 0
  }
}
```

A scale of 1.5 turns a 1024x1024 base into 1536x1536. `"steps": 0` reuses the
base render's step count. Denoise 0.3 blends the upscale without repainting it;
higher values smear faces even with a model-backed upscaler. Direct 2048x2048
generation is an option, but a RealESRGAN hires pass is sharper for the same
output size. Without an upscaler model, the built-in latent modes repaint the
image and are not a safe default.

## Timings

Measured on the Spark with the server defaults:

| Output | Time |
|---|---|
| 512x512, 12 steps | ~4 s |
| 1024x1024, 24 steps | ~27 s |
| 832x1216, 24 steps | ~27 s |
| 1248x1824 hires from 832x1216 | ~128 s |
| 512x512 edit, 24 steps | ~11 s |

These are render times. The client adds about 1 s for prompt expansion. Timings
vary with prompt length, queue depth and the desktop's other work.

## Sampling settings

The distilled model still needs enough steps for faces and hands. At 832x1216
and 1024x1024, 8 steps regularly produced mangled hands, distorted eyes and waxy
skin; 24 steps was clean, and the extra time is about 2.5x. `dpm++2m` with
`karras` is the default. EasyCache is disabled because it adds run-to-run drift
without a useful speedup at this size.

Keep the default 24 steps. Drop to 8-12 only for layout drafts you will discard,
and prefer a RealESRGAN hires pass over raising steps past 24 for detail.

| Field | What it does |
|---|---|
| `prompt` | The scene to draw. The client expands it into a caption first. |
| `negative_prompt` | Accepted, but inert while the service runs `--cfg-scale 1.0` (distilled guidance). Negatives need CFG > 1. |
| `width`, `height` | Output size. 1024x1024 is the native size. |
| `sample_params.sample_steps` | Sampling steps. The service default is 24. |
| `sample_params.sample_method` | Sampler, for example `dpm++2m` or `euler`. |
| `sample_params.scheduler` | Sigma schedule, for example `karras`. |
| `seed` | Top-level integer. Reproduces the composition; pixels can still vary. |
| `init_image` | Base64 source image for img2img. |
| `strength` | How much of the init image is repainted. |
| `ref_images` | Base64 reference images for conditioning, up to 10. |
| `hires` | Enables the second pass. |

## Add-ons

Put FLUX-compatible LoRA files in `models/flux3/loras`. They appear in
`GET /sdapi/v1/loras` and attach through the structured `lora` field; `<lora:...>`
tags in the prompt are ignored.

Put ESRGAN-family models in `models/flux3/upscalers` and name one in
`hires.upscaler` instead of using a built-in latent mode.

## Troubleshooting

- A JSON response with no image means the request failed. Print the full body
  and check `error`.
- Connection refused means a service is not running. Start the vision profile
  with `./sparkstack.sh up vision`, and the LLM profile with
  `./sparkstack.sh up llm` for prompt expansion. Check the cards at `:9190`.
- A native job that stays queued is waiting behind another render. Poll the job
  id and check `./sparkstack.sh logs image-flux3`.
- An edit that returns an unrecognizable subject usually had too much strength.
  Try 0.6 and name the subject in the prompt.
- Text in images is unreliable. Keep lettering out of the prompt; negative
  prompts do not take effect at `--cfg-scale 1.0`.

## Python client

The client has three commands. Run `scripts/flux_client.py <command> --help` for
the full flag list.

| Command | Use |
|---|---|
| `generate` | Text to image. Accepts `--width`, `--height`, `--steps`, `--sampler`, `--scheduler`, `--seed`, `--negative` and `--out`. |
| `edit` | Image to image. Pass the init image first, then any references (references are sent as `ref_images`, the init is not duplicated). Accepts `--strength`, `--width` and `--height`. |
| `hires` | Two-pass render. Accepts `--scale`, `--denoise`, `--hires-steps`, `--upscaler`, `--width` and `--height`. |

All three accept `--api`, `--chat-api`, `--upsample-model`, `--no-expand`,
`--show-prompt`, `--timeout`, `--seed`, `--steps`, `--sampler`, `--scheduler`,
`--negative` and `--out`. `edit` uses the first image's dimensions unless you pass `--width` and
`--height`.

The service definition lives in `docker-compose.yaml`. Stack setup and
monitoring are covered in the [README](../README.md).
