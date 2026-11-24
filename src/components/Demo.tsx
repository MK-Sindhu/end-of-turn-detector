"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { site } from "@/lib/site";
import { Live, Orb, OrbState } from "./Orb";

const RATE = 16000; // the model expects 16 kHz
const FRAME = 320; // 20 ms
const MIN_SILENCE_MS = 200; // ask the model after this much quiet
const FALLBACK_MS = 1500; // stop waiting and treat the turn as over
const MAX_SAMPLES = 8 * RATE;
const BASELINE_MS = 700; // the fixed timer shown for comparison

type Kind = "" | "wait" | "reply" | "cut";
type Row = {
  id: number;
  pause: string;
  said: string;
  decision: string;
  decisionKind: Kind;
  did: string;
  timer: string;
  timerKind: Kind;
};
type Result = { probability: number; complete: boolean; inference_ms: number };

const SAMPLES = [
  { file: "/samples/pause.wav", label: "A speaker pausing mid-sentence" },
  { file: "/samples/end.wav", label: "A speaker finishing a sentence" },
];

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
}
registerProcessor("tap", Tap);`;

function resample(samples: Float32Array, fromRate: number): Float32Array {
  if (fromRate === RATE) return samples;
  const out = new Float32Array(Math.floor((samples.length * RATE) / fromRate));
  for (let i = 0; i < out.length; i++) {
    const x = (i * fromRate) / RATE;
    const a = Math.floor(x);
    const b = Math.min(a + 1, samples.length - 1);
    out[i] = samples[a] + (samples[b] - samples[a]) * (x - a);
  }
  return out;
}

async function request(body: BodyInit, headers?: HeadersInit): Promise<Result> {
  const response = await fetch(`${site.api}/predict`, { method: "POST", body, headers });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail ?? response.statusText);
  return data;
}

export function Demo() {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState({ text: "The microphone is off.", kind: "" as Kind });
  const [rows, setRows] = useState<Row[]>([]);
  const [probability, setProbability] = useState<number | null>(null);
  const [silence, setSilence] = useState<number | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [sample, setSample] = useState<Record<string, string>>({});

  const live = useRef<Live>({ level: 0, state: "idle" });
  const thresholdRef = useRef(threshold);
  const audio = useRef<{ context: AudioContext; stream: MediaStream } | null>(null);
  // Everything about the turn in progress. Kept outside React state because it
  // changes fifty times a second.
  const turn = useRef({
    pending: new Float32Array(0),
    frames: [] as Float32Array[],
    floorDb: -60,
    speechMs: 0,
    silenceMs: 0,
    asked: false,
    decided: false,
    rowId: 0,
    replied: false,
    requestId: 0,
    nextRow: 1,
  });

  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  const patch = useCallback((id: number, change: Partial<Row>) => {
    setRows((all) => all.map((row) => (row.id === id ? { ...row, ...change } : row)));
  }, []);

  const say = useCallback((text: string, kind: Kind, state: OrbState) => {
    setStatus({ text, kind });
    live.current.state = state;
  }, []);

  const resetTurn = useCallback(() => {
    const t = turn.current;
    t.frames = [];
    t.speechMs = 0;
    t.silenceMs = 0;
    t.asked = false;
    t.decided = false;
    t.rowId = 0;
    t.replied = false;
    t.requestId++;
  }, []);

  const askModel = useCallback(async () => {
    const t = turn.current;
    t.asked = true;
    const requestId = ++t.requestId;
    const rowId = (t.rowId = t.nextRow++);
    t.replied = false;
    setRows((all) =>
      [
        { id: rowId, pause: "…", said: "…", decision: "", decisionKind: "" as Kind, did: "", timer: "", timerKind: "" as Kind },
        ...all,
      ].slice(0, 8),
    );
    const flat = new Float32Array(t.frames.length * FRAME);
    t.frames.forEach((frame, i) => flat.set(frame, i * FRAME));
    try {
      const result = await request(flat.slice(-MAX_SAMPLES).buffer as ArrayBuffer, {
        "Content-Type": "application/octet-stream",
      });
      const percent = `${Math.round(result.probability * 100)}%`;
      setProbability(result.probability);
      setLatency(result.inference_ms);
      patch(rowId, { said: `${percent} finished` });
      if (requestId !== t.requestId) return; // the speaker has already carried on
      if (result.probability >= thresholdRef.current) {
        t.decided = true;
        t.replied = true;
        patch(rowId, { decision: "Reply now", decisionKind: "reply" });
        say(`Finished. An agent would reply now, ${Math.round(t.silenceMs)} ms after you stopped.`, "reply", "reply");
      } else {
        patch(rowId, { decision: "Keep waiting", decisionKind: "wait" });
        say("A pause. The model thinks you have more to say.", "wait", "wait");
      }
    } catch (error) {
      patch(rowId, { said: "No answer" });
      say(`The model could not be reached (${(error as Error).message}). Check that the service is running.`, "cut", "cut");
    }
  }, [patch, say]);

  const closePause = useCallback(
    (resumed: boolean) => {
      const t = turn.current;
      if (!t.rowId) return;
      const pause = Math.round(t.silenceMs);
      if (resumed) {
        patch(t.rowId, {
          pause: `${pause} ms`,
          did: "Kept talking",
          timer: pause >= BASELINE_MS ? "Would have cut in" : "Would have waited",
          timerKind: pause >= BASELINE_MS ? "cut" : "",
          ...(t.replied ? { decision: "Replied too early", decisionKind: "cut" as Kind } : {}),
        });
      } else {
        patch(t.rowId, { pause: `${pause} ms`, did: "Had finished", timer: `Replies at ${BASELINE_MS} ms` });
      }
    },
    [patch],
  );

  const onFrame = useCallback(
    (frame: Float32Array) => {
      const t = turn.current;
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      const db = 10 * Math.log10(sum / frame.length + 1e-10);
      // The floor drops quickly to quiet levels and creeps up slowly, so it tracks the room.
      t.floorDb = db < t.floorDb ? t.floorDb * 0.9 + db * 0.1 : Math.min(t.floorDb + 0.02, -30);
      const speaking = db > t.floorDb + 12 && db > -55;
      live.current.level = speaking ? Math.max(0, Math.min(1, (db + 55) / 40)) : 0;

      if (speaking) {
        if (t.silenceMs >= MIN_SILENCE_MS && t.speechMs >= 300) closePause(true);
        if (t.decided) resetTurn();
        if (t.silenceMs > 0) {
          t.requestId++;
          t.asked = false;
          t.rowId = 0;
        }
        t.silenceMs = 0;
        t.speechMs += 20;
        t.frames.push(frame);
        if (t.frames.length > MAX_SAMPLES / FRAME) t.frames.shift();
        if (t.speechMs === 300) {
          say("Listening…", "", "listening");
          setSilence(0);
        }
        return;
      }
      if (t.speechMs < 300) {
        if (t.speechMs > 0 && (t.silenceMs += 20) > 400) resetTurn();
        return;
      }
      t.silenceMs += 20;
      if (!t.decided) t.frames.push(frame);
      if (t.silenceMs % 100 === 0) setSilence(t.silenceMs);
      if (!t.asked && t.silenceMs >= MIN_SILENCE_MS) void askModel();
      if (t.silenceMs >= FALLBACK_MS) {
        if (!t.decided) {
          t.decided = true;
          if (t.rowId) patch(t.rowId, { decision: `Reply (waited ${FALLBACK_MS} ms)`, decisionKind: "wait" });
          say(`Finished. The model was unsure, so the agent waited the full ${FALLBACK_MS} ms.`, "wait", "reply");
        }
        if (t.rowId) {
          closePause(false);
          t.rowId = 0;
        }
      }
    },
    [askModel, closePause, patch, resetTurn, say],
  );

  const stop = useCallback(() => {
    audio.current?.stream.getTracks().forEach((track) => track.stop());
    void audio.current?.context.close();
    audio.current = null;
    live.current = { level: 0, state: "idle" };
    setRunning(false);
    setSilence(null);
    setStatus({ text: "The microphone is off.", kind: "" });
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices || !window.AudioWorkletNode) {
      say("This browser cannot record here. Open the page over https in a current browser.", "cut", "cut");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      say("Microphone access was refused. Allow it in the browser's site settings, then start again.", "cut", "cut");
      return;
    }
    let context: AudioContext;
    try {
      context = new AudioContext({ sampleRate: RATE });
    } catch {
      context = new AudioContext();
    }
    const workletUrl = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await context.audioWorklet.addModule(workletUrl);
    const tap = new AudioWorkletNode(context, "tap");
    tap.port.onmessage = (event: MessageEvent<Float32Array>) => {
      const t = turn.current;
      const fresh = resample(event.data, context.sampleRate);
      const joined = new Float32Array(t.pending.length + fresh.length);
      joined.set(t.pending);
      joined.set(fresh, t.pending.length);
      let offset = 0;
      for (; offset + FRAME <= joined.length; offset += FRAME) onFrame(joined.slice(offset, offset + FRAME));
      t.pending = joined.slice(offset);
    };
    context.createMediaStreamSource(stream).connect(tap);
    audio.current = { context, stream };
    turn.current.pending = new Float32Array(0);
    turn.current.floorDb = -60;
    resetTurn();
    setRunning(true);
    say("Listening. Say a sentence and pause partway through.", "", "listening");
  }, [onFrame, resetTurn, say]);

  useEffect(() => stop, [stop]);

  const scoreSample = useCallback(async (file: string) => {
    setSample((all) => ({ ...all, [file]: "Scoring…" }));
    void new Audio(file).play().catch(() => {});
    try {
      const form = new FormData();
      form.append("file", await (await fetch(file)).blob(), "clip.wav");
      const result = await request(form);
      setProbability(result.probability);
      setLatency(result.inference_ms);
      live.current.state = result.complete ? "reply" : "wait";
      setSample((all) => ({
        ...all,
        [file]: `${Math.round(result.probability * 100)}% finished: ${result.complete ? "reply" : "keep waiting"}`,
      }));
    } catch (error) {
      setSample((all) => ({ ...all, [file]: `Could not be scored: ${(error as Error).message}` }));
    }
  }, []);

  return (
    <div className="demo glass">
      <div className="demo-stage">
        <Orb live={live} />
        <p className={`demo-status ${status.kind}`} aria-live="polite">
          {status.text}
        </p>
        <button className={`pill ${running ? "pill-ghost" : "pill-light"}`} onClick={running ? stop : start}>
          {running ? "Stop microphone" : "Start microphone"}
        </button>
      </div>

      <div className="demo-side">
        <dl className="readout">
          <div>
            <dt>Finished, says the model</dt>
            <dd>{probability === null ? "–" : `${Math.round(probability * 100)}%`}</dd>
          </div>
          <div>
            <dt>Current silence</dt>
            <dd>{silence === null ? "–" : `${silence} ms`}</dd>
          </div>
          <div>
            <dt>Time to score</dt>
            <dd>{latency === null ? "–" : `${latency} ms`}</dd>
          </div>
        </dl>

        <label className="slider">
          <span>
            Reply when the model is at least <b>{Math.round(threshold * 100)}%</b> sure
          </span>
          <input
            type="range"
            min={0.3}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(event) => setThreshold(parseFloat(event.target.value))}
          />
        </label>

        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Pause</th>
                <th>Model said</th>
                <th>Decision</th>
                <th>You</th>
                <th>Fixed 700 ms timer</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="empty">
                    Start the microphone and speak. Each pause adds a row here.
                  </td>
                </tr>
              )}
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>{row.pause}</td>
                  <td>{row.said}</td>
                  <td className={row.decisionKind}>{row.decision}</td>
                  <td>{row.did}</td>
                  <td className={row.timerKind}>{row.timer}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="samples">
          <p>No microphone? Score a real test recording:</p>
          {SAMPLES.map((item) => (
            <button key={item.file} className="sample" onClick={() => scoreSample(item.file)}>
              <span aria-hidden="true">▶</span> {item.label}
              {sample[item.file] && <em>{sample[item.file]}</em>}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
