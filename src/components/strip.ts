// Shared drawing for the "speech strip": speech as vertical bars, silence as
// gaps, and a tall mark wherever the model made a decision.

export type MarkKind = "wait" | "reply" | "cut";
export type Mark = { at: number; kind: MarkKind; text: string };

export function palette(element: Element) {
  const css = getComputedStyle(element);
  const read = (name: string) => css.getPropertyValue(name).trim();
  return {
    bar: read("--strip-bar"),
    quiet: read("--strip-quiet"),
    wait: read("--wait"),
    reply: read("--reply"),
    cut: read("--cut"),
    mono: read("--font-mono") || "monospace",
  };
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
  const step = width / slots;
  const mid = height / 2 + 8;
  const tallest = height - 56;
  g.clearRect(0, 0, width, height);
  for (let i = 0; i < levels.length; i++) {
    const x = i * step;
    if (levels[i] > 0) {
      const h = Math.max(4, levels[i] * tallest);
      g.fillStyle = colors.bar;
      g.fillRect(x, mid - h / 2, Math.max(1.5, step * 0.55), h);
    } else {
      g.fillStyle = colors.quiet;
      g.fillRect(x, mid - 1, Math.max(1.5, step * 0.55), 2);
    }
  }
  // Slots not reached yet are drawn as a faint baseline, so the strip never looks empty.
  g.fillStyle = colors.quiet;
  for (let i = levels.length; i < slots; i++) g.fillRect(i * step, mid - 1, Math.max(1.5, step * 0.55), 2);
  g.font = `500 12px ${colors.mono}`;
  g.textBaseline = "top";
  for (const mark of marks) {
    const x = mark.at * step;
    if (x < 0 || x > width) continue;
    g.fillStyle = colors[mark.kind];
    // The mark is drawn like a danda: one tall stroke.
    g.fillRect(x, 22, 3, height - 30);
    const label = g.measureText(mark.text).width;
    g.fillText(mark.text, Math.min(x, width - label - 2), 4);
  }
}
