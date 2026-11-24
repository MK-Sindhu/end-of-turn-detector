# Company website

Next.js site for the voice AI company: a home page and the end-of-turn detector's product page with a live demo.

## Run it

```bash
npm install
npm run dev        # http://localhost:3000
```

The live demo calls the detector's API. Start that service from the repository root (one folder up):

```bash
cd ..
uv run uvicorn indic_eot.service.app:app --port 7861
```

## Settings

- `NEXT_PUBLIC_EOT_API`: address of the detector API. Defaults to `http://localhost:7861`. Set it to the deployed service when hosting the site.
- `src/lib/site.ts`: company name, tagline, and the measured results shown on the product page. Update the numbers there when the model is retrained.

## Deploy

The site is static and has no server code of its own. On Vercel: import the repository, set **Root Directory** to `website`, and add `NEXT_PUBLIC_EOT_API` as an environment variable.
