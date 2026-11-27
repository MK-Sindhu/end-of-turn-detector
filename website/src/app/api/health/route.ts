// GET /api/health: confirms the model loads on this server.
import { MODEL_FILE, warm } from "@/lib/server/detector";


export async function GET() {
  const headers = { "Access-Control-Allow-Origin": "*" };
  try {
    await warm();
    return Response.json({ status: "ok", model: MODEL_FILE }, { headers });
  } catch {
    return Response.json({ status: "model not loaded" }, { status: 503, headers });
  }
}
