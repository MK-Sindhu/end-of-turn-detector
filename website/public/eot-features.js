// The audio features the end-of-turn model expects, in plain JavaScript.
//
// Used in two places: the demo's worker (public/eot-worker.js) loads it in the
// browser, and the API route (src/app/api/predict) imports it on the server.
// It mirrors src/indic_eot/features.py in the detector project: 25 ms windows
// every 10 ms, power spectrum, 80 mel bands, log, rescale.

(() => {
  const SAMPLE_RATE = 16000;
  const N_FFT = 400;
  const HOP = 160;
  const BINS = N_FFT / 2 + 1;
  const MELS = 80;
  const MAX_SECONDS = 8;
  const SAMPLES = MAX_SECONDS * SAMPLE_RATE;
  const MEL_FRAMES = SAMPLES / HOP; // 800
  const MASK_FRAMES = MEL_FRAMES / 2; // the encoder halves the frame rate

  const WINDOW = new Float64Array(N_FFT);
  const COS = new Float64Array(N_FFT);
  const SIN = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    WINDOW[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
    COS[n] = Math.cos((2 * Math.PI * n) / N_FFT);
    SIN[n] = Math.sin((2 * Math.PI * n) / N_FFT);
  }

  // Keep the last 8 seconds and pad shorter audio with silence at the start.
  // Returns the mel input [80 x 800] and the mask [400] of frames with real audio.
  function prepare(audio, filters) {
    const kept = audio.length > SAMPLES ? audio.subarray(audio.length - SAMPLES) : audio;
    const start = SAMPLES - kept.length;
    const half = N_FFT / 2;
    // Reflect-pad by half a window at both ends, as the Whisper extractor does.
    const padded = new Float64Array(SAMPLES + N_FFT);
    const at = (i) => (i < start ? 0 : kept[i - start]);
    for (let i = 0; i < SAMPLES; i++) padded[half + i] = at(i);
    for (let i = 0; i < half; i++) {
      padded[half - 1 - i] = at(i + 1);
      padded[half + SAMPLES + i] = at(SAMPLES - 2 - i);
    }

    const log = new Float32Array(MELS * MEL_FRAMES);
    const frame = new Float64Array(N_FFT);
    const power = new Float64Array(BINS);
    let highest = -10;
    for (let t = 0; t < MEL_FRAMES; t++) {
      const offset = t * HOP;
      let silent = true;
      for (let n = 0; n < N_FFT; n++) {
        const value = padded[offset + n];
        if (value !== 0) silent = false;
        frame[n] = value * WINDOW[n];
      }
      if (silent) {
        for (let m = 0; m < MELS; m++) log[m * MEL_FRAMES + t] = -10;
        continue;
      }
      for (let k = 0; k < BINS; k++) {
        let re = 0;
        let im = 0;
        let phase = 0;
        for (let n = 0; n < N_FFT; n++) {
          re += frame[n] * COS[phase];
          im -= frame[n] * SIN[phase];
          phase += k;
          if (phase >= N_FFT) phase -= N_FFT;
        }
        power[k] = re * re + im * im;
      }
      for (let m = 0; m < MELS; m++) {
        let sum = 0;
        for (let k = 0; k < BINS; k++) sum += power[k] * filters[k * MELS + m];
        const value = Math.log10(Math.max(sum, 1e-10));
        log[m * MEL_FRAMES + t] = value;
        if (value > highest) highest = value;
      }
    }
    const lowest = highest - 8;
    for (let i = 0; i < log.length; i++) log[i] = (Math.max(log[i], lowest) + 4) / 4;

    const mask = new Uint8Array(MASK_FRAMES);
    const real = Math.max(1, Math.ceil((kept.length / SAMPLE_RATE) * 50));
    mask.fill(1, Math.max(0, MASK_FRAMES - real));
    return { mel: log, mask };
  }

  const eotFeatures = { prepare, SAMPLE_RATE, MELS, MEL_FRAMES, MASK_FRAMES };
  if (typeof module !== "undefined") module.exports = eotFeatures;
  else self.eotFeatures = eotFeatures;
})();
