// Runs the end-of-turn model inside the visitor's browser, off the main thread.
//
// The page posts { id, audio } where audio is a Float32Array of 16 kHz mono
// samples. The worker answers { id, probability, ms } or { id, error }.

importScripts("/eot-features.js", "/ort/ort.wasm.min.js");
const { prepare, MELS, MEL_FRAMES, MASK_FRAMES } = self.eotFeatures;
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;

const ready = (async () => {
  const [filters, session] = await Promise.all([
    fetch("/model/mel_filters.bin").then((r) => {
      if (!r.ok) throw new Error("the mel bands did not download");
      return r.arrayBuffer();
    }),
    ort.InferenceSession.create("/model/audio_eot.int8.onnx", { executionProviders: ["wasm"] }),
  ]);
  return { filters: new Float32Array(filters), session };
})();

ready.then(
  () => self.postMessage({ ready: true }),
  (error) => self.postMessage({ ready: false, error: String(error.message || error) }),
);

self.onmessage = async (event) => {
  const { id, audio } = event.data;
  try {
    const { filters, session } = await ready;
    const started = performance.now();
    const { mel, mask } = prepare(audio, filters);
    const output = await session.run({
      mel: new ort.Tensor("float32", mel, [1, MELS, MEL_FRAMES]),
      mask: new ort.Tensor("bool", mask, [1, MASK_FRAMES]),
    });
    const probability = output.probability.data[0];
    self.postMessage({ id, probability, ms: Math.round(performance.now() - started) });
  } catch (error) {
    self.postMessage({ id, error: String(error.message || error) });
  }
};
