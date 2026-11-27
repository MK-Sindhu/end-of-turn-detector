// POST /api/predict: send the speech so far, get the probability that the
// speaker has finished. Two ways to send audio:
//   - a WAV file in a form field named "file"
//   - raw 32-bit float samples, 16 kHz mono, as application/octet-stream
import { score } from "@/lib/server/detector";
import { decodeWav } from "@/lib/server/wav";

export const maxDuration = 30;

const RATE = 16000;
const MAX_BYTES = 4 * 1024 * 1024; // hosting caps request bodies near this size
const MIN_SECONDS = 0.1;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function reply(body: unknown, status = 200) {
  return Response.json(body, { status, headers: CORS });
}

export function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS });
}

export async function POST(request: Request) {
  if (Number(request.headers.get("content-length") ?? 0) > MAX_BYTES) {
    return reply({ detail: "The audio is too large. The limit is 4 MB; only the last 8 seconds are used." }, 413);
  }

  let audio: Float32Array;
  try {
    const type = request.headers.get("content-type") ?? "";
    if (type.startsWith("multipart/form-data")) {
      const file = (await request.formData()).get("file");
      if (!(file instanceof Blob)) return reply({ detail: 'Attach the recording as a form field named "file".' }, 400);
      if (file.size > MAX_BYTES) return reply({ detail: "The audio is too large. The limit is 4 MB." }, 413);
      audio = decodeWav(new Uint8Array(await file.arrayBuffer()));
    } else if (type.startsWith("application/octet-stream")) {
      const bytes = await request.arrayBuffer();
      if (bytes.byteLength > MAX_BYTES) return reply({ detail: "The audio is too large. The limit is 4 MB." }, 413);
      if (bytes.byteLength % 4 !== 0) {
        return reply({ detail: "Raw audio must be 32-bit float samples, so its size must be a multiple of 4 bytes." }, 400);
      }
      audio = new Float32Array(bytes);
    } else {
      return reply(
        { detail: 'Send a WAV file as a form field named "file", or raw float32 samples as application/octet-stream.' },
        415,
      );
    }
  } catch (error) {
    return reply({ detail: (error as Error).message }, 400);
  }

  if (audio.length < MIN_SECONDS * RATE) {
    return reply({ detail: "The audio is shorter than 0.1 seconds." }, 400);
  }
  if (!audio.every(Number.isFinite)) {
    return reply({ detail: "The audio contains values that are not numbers." }, 400);
  }

  try {
    const { probability, ms } = await score(audio);
    return reply({
      probability: Math.round(probability * 10000) / 10000,
      complete: probability >= 0.5,
      seconds_received: Math.round((audio.length / RATE) * 100) / 100,
      inference_ms: Math.round(ms * 10) / 10,
    });
  } catch {
    return reply({ detail: "The model could not score this audio." }, 500);
  }
}
