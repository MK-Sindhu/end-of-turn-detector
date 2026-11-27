import type { NextConfig } from "next";

// Files the API routes read at run time. They are not imported in code, so
// the build has to be told to ship them with the server functions.
const modelFiles = [
  "./public/model/audio_eot.int8.onnx",
  "./public/model/mel_filters.bin",
  "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
  "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // Load onnxruntime-web from node_modules on the server instead of bundling it.
  serverExternalPackages: ["onnxruntime-web"],
  outputFileTracingIncludes: {
    "/api/predict": modelFiles,
    "/api/health": modelFiles,
  },
};

export default nextConfig;
