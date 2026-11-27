// Loads the model once per server instance and scores audio with it.
// The same model file and feature code as the in-browser demo.
import { readFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-web";
import features from "../../../public/eot-features.js";

const MODEL_DIR = path.join(process.cwd(), "public", "model");
export const MODEL_FILE = "audio_eot.int8.onnx";

let loaded: Promise<{ session: ort.InferenceSession; filters: Float32Array }> | null = null;

function load() {
  if (!loaded) {
    loaded = (async () => {
      ort.env.wasm.numThreads = 1;
      const [model, bands] = await Promise.all([
        readFile(path.join(MODEL_DIR, MODEL_FILE)),
        readFile(path.join(MODEL_DIR, "mel_filters.bin")),
      ]);
      const session = await ort.InferenceSession.create(new Uint8Array(model));
      const filters = new Float32Array(bands.buffer.slice(bands.byteOffset, bands.byteOffset + bands.byteLength));
      return { session, filters };
    })();
    loaded.catch(() => {
      loaded = null; // try again on the next request
    });
  }
  return loaded;
}

/** Probability (0 to 1) that the speaker has finished. audio: 16 kHz mono. */
export async function score(audio: Float32Array): Promise<{ probability: number; ms: number }> {
  const { session, filters } = await load();
  const started = performance.now();
  const { mel, mask } = features.prepare(audio, filters);
  const output = await session.run({
    mel: new ort.Tensor("float32", mel, [1, features.MELS, features.MEL_FRAMES]),
    mask: new ort.Tensor("bool", mask, [1, features.MASK_FRAMES]),
  });
  const probability = Number((output.probability.data as Float32Array)[0]);
  return { probability, ms: performance.now() - started };
}

export async function warm(): Promise<void> {
  await load();
}
