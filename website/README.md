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

## Settings

- `NEXT_PUBLIC_EOT_API`: address shown in the API examples on the product page. Optional. The demo does not use it.

## Deploying

The site is static and has no server code of its own. On Vercel: import the repository, set **Root Directory** to `website`. No environment variables are required.
