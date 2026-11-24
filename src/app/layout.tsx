import type { Metadata } from "next";
import { Anek_Devanagari, Geist_Mono, Schibsted_Grotesk } from "next/font/google";
import { site } from "@/lib/site";
import "./globals.css";

const sans = Schibsted_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });
const hindi = Anek_Devanagari({ subsets: ["devanagari", "latin"], variable: "--font-hindi" });

export const metadata: Metadata = {
  title: { default: `${site.company}: ${site.tagline}`, template: `%s · ${site.company}` },
  description:
    "Models and tools for voice agents that take turns the way people do. End-of-turn detection for Indian speech, with a live demo and measured results.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${hindi.variable}`}>
      <body>{children}</body>
    </html>
  );
}
