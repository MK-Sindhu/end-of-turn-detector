# Company website

Next.js site for the voice AI company: a home page and the end-of-turn detector's product page with a live demo.

## Run it

```bash
npm install
npm run dev        # http://localhost:3000
```

The live demo runs the model inside the visitor's browser, so the site needs no model server:

- `public/eot-worker.js` computes the audio features and runs the model with onnxruntime-web.
- `public/model/` holds the 10 MB model. It is written by `uv run dvc repro site_model` in the repository root; rerun that after retraining and commit the result.
- `public/ort/` holds the onnxruntime-web files. `npm run dev` and `npm run build` copy them from `node_modules`, so they are not committed.

## The public API

The site also serves the model over HTTP, from `src/app/api`:

- `POST /api/predict`: send a WAV file as a form field named `file`, or raw 32-bit float samples at 16 kHz as `application/octet-stream`. Returns `probability`, `complete`, `seconds_received` and `inference_ms`. Limit 4 MB.
- `GET /api/health`: confirms the model loads.

```bash
curl -X POST http://localhost:3000/api/predict -F "file=@public/samples/pause.wav"
```

It is open (no key, no rate limit) and allows calls from any website. It uses the same model file and the same feature code (`public/eot-features.js`) as the demo.

## Settings

- `NEXT_PUBLIC_EOT_API`: address shown in the API examples on the product page. Optional. On Vercel it is worked out from the project's production domain.

## Deploying

The site is static and has no server code of its own. On Vercel: import the repository, set **Root Directory** to `website`. No environment variables are required.
