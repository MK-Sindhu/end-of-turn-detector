// Shared drawing for the "speech stage": the voice rises from a dark slab as
// a ribbon of fine spectral lines. Words are peaks, silence lies flat, and a
// glass tag floats over each pause the model judged.

export type MarkKind = "wait" | "reply" | "cut";
export type Mark = { at: number; kind: MarkKind; text: string };

const LAYERS = 30;
const SPECTRUM = ["#ff7fa9", "#b574ff", "#5a8dff", "#35d6a0", "#b9e56c"];
const TAG: Record<MarkKind, string> = { wait: "#b86a00", reply: "#0b8a63", cut: "#d1432f" };

export function palette(element: Element) {
  const css = getComputedStyle(element);
  return { mono: css.getPropertyValue("--font-mono").trim() || "monospace" };
}

export function fit(canvas: HTMLCanvasElement) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  const g = canvas.getContext("2d")!;
  g.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { g, width: box.width, height: box.height };
}

function smooth(levels: number[], slots: number) {
  const out = new Array<number>(slots).fill(0);
  const radius = Math.max(2, Math.round(slots / 60));
  for (let i = 0; i < slots; i++) {
    let sum = 0;
    let weight = 0;
    for (let j = -radius; j <= radius; j++) {
      const w = radius + 1 - Math.abs(j);
      sum += (levels[i + j] ?? 0) * w;
      weight += w;
    }
    out[i] = sum / weight;
  }
  return out;
}

function pill(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number) {
  g.beginPath();
  g.roundRect(x, y, w, h, h / 2);
}

// levels: one value per slot, 0 for silence, up to 1 for loud speech.
export function drawStrip(
  g: CanvasRenderingContext2D,
  width: number,
  height: number,
  levels: number[],
  marks: Mark[],
  colors: ReturnType<typeof palette>,
  slots: number,
) {
  g.clearRect(0, 0, width, height);
  const skew = Math.min(90, width * 0.09); // how far the back edge shifts right
  const depth = Math.min(84, height * 0.24); // slab depth on screen
  const front = height - 22;
  const back = front - depth;
  const span = width - skew - 8;
  const tallest = back - 54;
  const envelope = smooth(levels, slots);

  // The slab: a dark device-like surface seen at an angle.
  g.beginPath();
  g.moveTo(skew, back);
  g.lineTo(width - 2, back);
  g.lineTo(width - skew - 2, front);
  g.lineTo(2, front);
  g.closePath();
  const surface = g.createLinearGradient(0, back, 0, front);
  surface.addColorStop(0, "#1d2b25");
  surface.addColorStop(1, "#0f1613");
  g.fillStyle = surface;
  g.shadowColor = "rgba(30, 50, 60, 0.35)";
  g.shadowBlur = 30;
  g.shadowOffsetY = 18;
  g.fill();
  g.shadowColor = "transparent";
  g.fillStyle = "#cfd8e3";
  g.fillRect(2, front, width - skew - 4, 5); // the lit front edge

  // The ribbon: the same curve repeated from back to front, shrinking as it
  // comes forward, which gives the nested arcs.
  const stroke = g.createLinearGradient(0, 0, width, 0);
  SPECTRUM.forEach((color, i) => stroke.addColorStop(i / (SPECTRUM.length - 1), color));
  g.strokeStyle = stroke;
  g.lineWidth = 1.15;
  g.lineJoin = "round";
  for (let k = 0; k < LAYERS; k++) {
    const t = k / (LAYERS - 1); // 0 at the back, 1 at the front
    const baseY = back + 6 + (depth - 12) * t;
    const shift = skew * (1 - t);
    const scale = 1 - 0.62 * t;
    g.globalAlpha = 0.35 + 0.6 * (1 - t);
    g.beginPath();
    for (let i = 0; i < slots; i++) {
      const x = shift + 4 + (i / (slots - 1)) * span;
      const y = baseY - envelope[i] * tallest * scale;
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.stroke();
  }
  g.globalAlpha = 1;

  // Glass tags over the pauses.
  g.font = `500 12px ${colors.mono}`;
  g.textBaseline = "middle";
  let lastRight = -Infinity;
  let row = 0;
  for (const mark of marks) {
    const x = skew * 0.5 + 4 + (mark.at / (slots - 1)) * span;
    if (x < 0 || x > width) continue;
    const w = g.measureText(mark.text).width + 34;
    const left = Math.max(4, Math.min(x - 16, width - w - 4));
    row = left < lastRight + 8 ? (row + 1) % 3 : 0;
    lastRight = left + w;
    const top = 8 + row * 34;
    g.strokeStyle = TAG[mark.kind];
    g.globalAlpha = 0.55;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(x, top + 26);
    g.lineTo(x, back + depth / 2);
    g.stroke();
    g.globalAlpha = 1;
    pill(g, left, top, w, 26);
    g.fillStyle = "rgba(255, 255, 255, 0.82)";
    g.shadowColor = "rgba(40, 60, 90, 0.18)";
    g.shadowBlur = 14;
    g.shadowOffsetY = 4;
    g.fill();
    g.shadowColor = "transparent";
    g.strokeStyle = "rgba(255, 255, 255, 0.95)";
    g.stroke();
    g.fillStyle = TAG[mark.kind];
    g.beginPath();
    g.arc(left + 14, top + 13, 4.5, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = "#14201b";
    g.fillText(mark.text, left + 25, top + 13.5);
  }
}
