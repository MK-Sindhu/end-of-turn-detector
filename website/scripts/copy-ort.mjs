// Copies the onnxruntime-web files the demo's worker loads into public/ort.
// Runs before `npm run dev` and `npm run build`, so they are never committed.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "onnxruntime-web", "dist");
const to = join(root, "public", "ort");

mkdirSync(to, { recursive: true });
for (const file of ["ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  copyFileSync(join(from, file), join(to, file));
}
