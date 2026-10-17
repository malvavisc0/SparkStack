# Image generation and editing

The `image-flux3` service runs FLUX.3 Image Klein on port `:9196`. It serves the
OpenAI image routes for normal work and a native async API for hires passes,
exact seeds and sampler control. Both routes use the same model and the same
defaults: 8 steps, `dpm++2m` with the `karras` scheduler, and strength 0.9 for
edits. A 1024x1024 generation takes about 11 s on the Spark; a 1536x1536 hires
render takes about 36 s.

## Quick start

`scripts/flux_client.py` is a command line client built on the Python standard
library. It needs Python 3.8 or newer and no packages.

```bash
./scripts/flux_client.py generate "a red fox in a snowy forest, evening light" --out fox.png
./scripts/flux_client.py edit "change the mug to bright blue" mug.png --out mug-blue.png
./scripts/flux_client.py hires "a red fox in a snowy forest, evening light" --scale 1.5 --out fox-large.png
```

The client talks to `http://localhost:9196` by default. Point it at another host
with `--api` or the `SPARKSTACK_IMAGE_URL` environment variable:

```bash
./scripts/flux_client.py generate "a red fox in a snowy forest" \
  --api http://192.168.1.10:9196 --out fox.png
```

It prints the output path, the image dimensions and the time the job took. Add
`--seed` to make a result repeatable.

## Generate

The OpenAI route is the shortest path for simple integrations. It returns JSON,
not image bytes, so decode `data[0].b64_json` before writing the file:

```bash
curl -s http://192.168.1.10:9196/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red fox in a snowy forest, evening light", "size": "1024x1024"}' \
  | python3 -c 'import json,sys,base64; open("fox.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

The server generates at 1024x1024 when `size` is absent. That is the model's
native size and the best detail per second. A direct 2048x2048 render works, but
it is softer than a hires pass and takes about 52 s.

The route accepts `<sd_cpp_extra_args>` in the prompt for per-request overrides.
Sampler fields go under `sample_params`:

```text
a red fox in a snowy forest <sd_cpp_extra_args>{"sample_params":{"sample_steps":16,"seed":42}}</sd_cpp_extra_args>
```

The tag is removed before the prompt reaches the model. Both routes return PNG
data by default. The native API below is easier to use when a client needs to
change several settings.

## Edit

`POST /v1/images/edits` uses multipart form data. Send one or more `image`
fields. The first image becomes the img2img init image, and every image is
passed to the model as a reference. The prompt should describe the finished
scene, not only the change.

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
change the mug to bright blue <sd_cpp_extra_args>{"strength":0.6}</sd_cpp_extra_args>
```

The native API takes `strength` as a normal JSON field. Send the source image as
a base64 string in `init_image` and any extra references in `ref_images`; the
Python client encodes both for you.

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
    "prompt": "a red fox in a snowy forest, evening light",
    "width": 1024,
    "height": 1024,
    "sample_params": {"sample_steps": 8, "sample_method": "dpm++2m", "scheduler": "karras", "seed": 42}
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
  "prompt": "a red fox in a snowy forest, evening light",
  "width": 1024,
  "height": 1024,
  "hires": {
    "enabled": true,
    "upscaler": "Latent (nearest-exact)",
    "scale": 1.5,
    "denoising_strength": 0.45,
    "steps": 8
  }
}
```

A scale of 1.5 turns a 1024x1024 base into 1536x1536. Denoise 0.45 keeps the
base composition; 0.55 repaints more if the upscale looks soft. Direct 2048x2048
generation is an option, but the hires pass is sharper for the same output size.

## Timings

Measured on the Spark with the server defaults:

| Output | Time |
|---|---|
| 512x512 generation | ~2.5 s |
| 1024x1024 generation | ~11 s |
| 1536x1536 hires from 1024 | ~36 s |
| 2048x2048 direct | ~52 s |
| 512x512 edit | ~5 s |

Timings vary with prompt length, queue depth and the desktop's other work.

## Sampling settings

This model is distilled, so more steps do not add detail. In a fixed-seed
comparison, 8 and 16 steps looked similar while 24 and 32 steps produced smoother
images at a higher cost. `dpm++2m` with `karras` produced more edge detail than
`euler` at the same step count. EasyCache made no measurable difference here.

Keep the default 8 steps for drafts and most final work. Raise the step count
only when you want a smoother result.

| Field | What it does |
|---|---|
| `prompt` | The scene to draw. Natural language works better than tag lists. |
| `negative_prompt` | Things to avoid, such as `text, letters, words, logos`. |
| `width`, `height` | Output size. 1024x1024 is the native size. |
| `sample_params.sample_steps` | Sampling steps. The service default is 8. |
| `sample_params.sample_method` | Sampler, for example `dpm++2m` or `euler`. |
| `sample_params.scheduler` | Sigma schedule, for example `karras`. |
| `sample_params.seed` | Fix this value to reproduce a result. |
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
- Connection refused means the service is not running. Start the vision profile
  with `./sparkstack.sh up vision` and check the card at `:9190`.
- A native job that stays queued is waiting behind another render. Poll the job
  id and check `./sparkstack.sh logs image-flux3`.
- An edit that returns an unrecognizable subject usually had too much strength.
  Try 0.6 and name the subject in the prompt.
- Text in images is unreliable. Keep lettering out of the prompt or add
  `text, letters, words` to `--negative`.

## Python client

The client has three commands. Run `scripts/flux_client.py <command> --help` for
the full flag list.

| Command | Use |
|---|---|
| `generate` | Text to image. Accepts `--width`, `--height`, `--steps`, `--sampler`, `--scheduler`, `--seed`, `--negative` and `--out`. |
| `edit` | Image to image. Pass the init image first, then any references. Accepts `--strength`, `--width` and `--height`. |
| `hires` | Two-pass render. Accepts `--scale`, `--denoise`, `--hires-steps`, `--upscaler`, `--width` and `--height`. |

All three accept `--api`, `--timeout`, `--seed`, `--steps`, `--sampler`,
`--scheduler`, `--negative` and `--out`. `edit` uses the first image's
dimensions unless you pass `--width` and `--height`.

The service definition lives in `docker-compose.yaml`. Stack setup and
monitoring are covered in the [README](../README.md).
