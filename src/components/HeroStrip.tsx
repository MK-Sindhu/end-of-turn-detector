"use client";

import { useEffect, useRef } from "react";
import { drawStrip, fit, Mark, palette } from "./strip";

const SLOTS = 150;

// One scripted turn, replayed: five words, a 900 ms pause, three more words,
// then the end. It shows the decision a voice agent faces at every silence.
function script() {
  const levels: number[] = [];
  const word = (bars: number, seed: number) => {
    for (let i = 0; i < bars; i++) {
      const shape = Math.sin(((i + 0.5) / bars) * Math.PI);
      levels.push(0.25 + 0.7 * shape * (0.6 + 0.4 * Math.abs(Math.sin(seed * 12.9 + i * 4.1))));
    }
    levels.push(0, 0);
  };
  [9, 6, 8, 5, 8].forEach(word);
  const pauseStart = levels.length;
  for (let i = 0; i < 40; i++) levels.push(0);
  const pauseEnd = levels.length;
  [7, 10, 12].forEach((bars, i) => word(bars, i + 7));
  const end = levels.length;
  while (levels.length < SLOTS) levels.push(0);
  return { levels, pauseStart, pauseEnd, end };
}

export function HeroStrip() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const { levels, pauseStart, pauseEnd, end } = script();
    const colors = palette(canvas);
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let size = fit(canvas);
    let frame = 0;
    const started = performance.now();

    const render = (now: number) => {
      // 45 slots per second, then hold the finished picture before replaying.
      const cursor = still ? SLOTS : Math.min(SLOTS, (((now - started) / 1000) % 9) * 45);
      const marks: Mark[] = [];
      if (cursor > pauseStart + 9) marks.push({ at: pauseStart + 9, kind: "wait", text: "9% finished · keep waiting" });
      if (cursor > pauseStart + 31) marks.push({ at: pauseStart + 31, kind: "cut", text: "a 700 ms timer cuts in here" });
      if (cursor > end + 9) marks.push({ at: end + 9, kind: "reply", text: "96% finished · reply" });
      void pauseEnd;
      drawStrip(size.g, size.width, size.height, levels.slice(0, Math.floor(cursor)), marks, colors, SLOTS);
      if (!still) frame = requestAnimationFrame(render);
    };
    const onResize = () => {
      size = fit(canvas);
      if (still) render(0);
    };
    window.addEventListener("resize", onResize);
    frame = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      className="strip strip-hero"
      role="img"
      aria-label="A spoken sentence drawn as bars. The speaker pauses for 900 milliseconds mid-sentence. The model waits; a fixed 700 millisecond timer would have interrupted. At the true end the model replies after 200 milliseconds."
    />
  );
}
