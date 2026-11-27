// Read a WAV file into 16 kHz mono samples. Handles the common encodings:
// 8, 16, 24 and 32-bit integers and 32 or 64-bit floats, any channel count.

const TARGET_RATE = 16000;

export function decodeWav(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (at: number) => String.fromCharCode(...bytes.subarray(at, at + 4));
  if (bytes.length < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
    throw new Error("The file is not a WAV recording.");
  }

  let format = 0;
  let channels = 0;
  let rate = 0;
  let bits = 0;
  let dataStart = -1;
  let dataLength = 0;
  for (let at = 12; at + 8 <= bytes.length; ) {
    const size = view.getUint32(at + 4, true);
    if (tag(at) === "fmt " && size >= 16) {
      format = view.getUint16(at + 8, true);
      channels = view.getUint16(at + 10, true);
      rate = view.getUint32(at + 12, true);
      bits = view.getUint16(at + 22, true);
      // "Extensible" files keep the real encoding further in.
      if (format === 0xfffe && size >= 26) format = view.getUint16(at + 32, true);
    } else if (tag(at) === "data") {
      dataStart = at + 8;
      dataLength = Math.min(size, bytes.length - dataStart);
      break;
    }
    at += 8 + size + (size % 2);
  }

  const float = format === 3 && (bits === 32 || bits === 64);
  const integer = format === 1 && [8, 16, 24, 32].includes(bits);
  if (dataStart < 0 || !channels || !rate || !(float || integer)) {
    throw new Error("This WAV encoding is not supported. Send 16-bit PCM or 32-bit float.");
  }

  const width = bits / 8;
  const count = Math.floor(dataLength / (width * channels));
  const mono = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const at = dataStart + (i * channels + c) * width;
      if (float) sum += bits === 32 ? view.getFloat32(at, true) : view.getFloat64(at, true);
      else if (bits === 8) sum += (view.getUint8(at) - 128) / 128;
      else if (bits === 16) sum += view.getInt16(at, true) / 32768;
      else if (bits === 24) sum += ((view.getInt8(at + 2) << 16) | view.getUint16(at, true)) / 8388608;
      else sum += view.getInt32(at, true) / 2147483648;
    }
    mono[i] = sum / channels;
  }
  return rate === TARGET_RATE ? mono : resample(mono, rate);
}

// Straight-line interpolation between neighbouring samples. Good enough for
// recordings made at 44.1 or 48 kHz; send 16 kHz audio to skip this step.
function resample(samples: Float32Array, fromRate: number): Float32Array {
  const out = new Float32Array(Math.floor((samples.length * TARGET_RATE) / fromRate));
  for (let i = 0; i < out.length; i++) {
    const x = (i * fromRate) / TARGET_RATE;
    const a = Math.floor(x);
    const b = Math.min(a + 1, samples.length - 1);
    out[i] = samples[a] + (samples[b] - samples[a]) * (x - a);
  }
  return out;
}
