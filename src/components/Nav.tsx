import Link from "next/link";
import { site } from "@/lib/site";

export function Nav() {
  return (
    <header className="nav">
      <Link href="/" className="brand">
        <span className="brand-mark" aria-hidden="true" />
        {site.company}
      </Link>
      <nav aria-label="Main">
        <Link href="/viraam">End-of-turn detector</Link>
        <Link href="/#arena">Stress-test arena</Link>
        <Link href="/viraam#api">API</Link>
      </nav>
      <Link href="/viraam#demo" className="pill pill-light">
        Try the demo
      </Link>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="footer">
      <div>
        <span className="brand">
          <span className="brand-mark" aria-hidden="true" />
          {site.company}
        </span>
        <p>Viraam (विराम) is Hindi for “pause”.</p>
      </div>
      <div className="footer-links">
        <Link href="/viraam">End-of-turn detector</Link>
        <Link href="/viraam#results">Results</Link>
        <Link href="/viraam#api">API</Link>
      </div>
      <p className="fine">Research preview. Every number on this site comes from a run recorded in the project pipeline.</p>
    </footer>
  );
}
