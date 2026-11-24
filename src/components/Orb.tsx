"use client";

import { useEffect, useRef } from "react";

export type OrbState = "idle" | "listening" | "wait" | "reply" | "cut";
export type Live = { level: number; state: OrbState };

const TINT: Record<OrbState, string> = {
  idle: "#8a8f98",
  listening: "#f3f1ec",
  wait: "#f2a84b",
  reply: "#7fb2e6",
  cut: "#e8572a",
};

// A liquid-metal sphere. Its outline ripples with the loudness of the voice
// and its tint shows what the model decided at the last pause.
export function Orb({ live }: { live: React.RefObject<Live> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const g = canvas.getContext("2d");
    if (!g) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const size = 420;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.height = size * ratio;
    let frame = 0;
    let level = 0;

    const draw = (ms: number) => {
      const t = still ? 0 : ms / 1000;
      const { level: target, state } = live.current ?? { level: 0, state: "idle" as OrbState };
      level += (target - level) * 0.18;
      const tint = TINT[state];
      const c = size / 2;
      const radius = 118 * (1 + level * 0.1);

      g.setTransform(ratio, 0, 0, ratio, 0, 0);
      g.clearRect(0, 0, size, size);

      const outline = new Path2D();
      for (let i = 0; i <= 120; i++) {
        const a = (i / 120) * Math.PI * 2;
        const wobble =
          1 +
          (0.012 + level * 0.07) * Math.sin(3 * a + t * 1.7) +
          (0.008 + level * 0.05) * Math.sin(5 * a - t * 2.3);
        const x = c + Math.cos(a) * radius * wobble;
        const y = c + Math.sin(a) * radius * wobble;
        if (i === 0) outline.moveTo(x, y);
        else outline.lineTo(x, y);
      }

      // Glow behind the sphere, in the colour of the current decision.
      g.save();
      g.shadowColor = tint;
      g.shadowBlur = 60 + level * 60;
      g.globalAlpha = state === "idle" ? 0.25 : 0.55;
      g.fillStyle = tint;
      g.fill(outline);
      g.restore();

      g.save();
      g.clip(outline);
      const base = g.createRadialGradient(c - 46, c - 56, 8, c, c, radius * 1.25);
      base.addColorStop(0, "#ffffff");
      base.addColorStop(0.35, "#b9bcc2");
      base.addColorStop(0.7, "#3b3d42");
      base.addColorStop(1, "#08080a");
      g.fillStyle = base;
      g.fillRect(0, 0, size, size);

      // Drifting highlights and shadows give the metal its liquid look.
      const blobs = [
        { r: 0.55, speed: 0.5, phase: 0, light: true },
        { r: 0.45, speed: -0.35, phase: 2.1, light: false },
        { r: 0.35, speed: 0.8, phase: 4.2, light: true },
        { r: 0.5, speed: -0.6, phase: 1.1, light: false },
      ];
      for (const blob of blobs) {
        const a = t * blob.speed + blob.phase;
        const x = c + Math.cos(a) * radius * 0.5;
        const y = c + Math.sin(a * 1.3) * radius * 0.45;
        const spot = g.createRadialGradient(x, y, 0, x, y, radius * blob.r);
        spot.addColorStop(0, blob.light ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.6)");
        spot.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = spot;
        g.fillRect(0, 0, size, size);
      }
      g.globalCompositeOperation = "overlay";
      g.globalAlpha = state === "idle" || state === "listening" ? 0.15 : 0.7;
      g.fillStyle = tint;
      g.fillRect(0, 0, size, size);
      g.restore();

      if (!still) frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [live]);

  return <canvas ref={canvasRef} className="orb" aria-hidden="true" />;
}
