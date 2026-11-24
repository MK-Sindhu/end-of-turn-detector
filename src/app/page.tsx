import Link from "next/link";
import { HeroStrip } from "@/components/HeroStrip";
import { Footer, Nav } from "@/components/Nav";
import { site } from "@/lib/site";

export default function Home() {
  return (
    <>
      <section className="hero hero-home">
        <Nav />
        <div className="hero-body">
          <p className="eyebrow">Voice AI, built for how India speaks</p>
          <h1>
            Voice agents that know when you have finished<span className="danda" aria-hidden="true">।</span>
          </h1>
          <p className="lede">
            People pause to think. Most voice agents hear silence and jump in. {site.company} builds the
            models and the tests that teach agents to wait for the end of the thought.
          </p>
          <div className="actions">
            <Link href="/viraam#demo" className="btn btn-solid">Try it with your voice</Link>
            <Link href="/viraam#results" className="btn btn-line">See the measurements</Link>
          </div>
        </div>
        <figure className="hero-figure">
          <HeroStrip />
          <figcaption>
            <span lang="hi">मुझे कल शाम के लिए … दो टिकट चाहिए।</span>
            “For tomorrow evening … I need two tickets.” One turn, with the pause every agent has to judge.
          </figcaption>
        </figure>
      </section>

      <main>
        <section className="section" id="products">
          <p className="eyebrow">Two projects</p>
          <h2>One teaches the agent to listen. The other tries to break it<span className="danda" aria-hidden="true">।</span></h2>
          <div className="products">
            <Link href="/viraam" className="product card">
              <span className="tag tag-live">Live demo</span>
              <h3>End-of-turn detector</h3>
              <p>
                A 10 MB model that hears a pause and answers one question: has the speaker finished, or
                are they still thinking? Trained on Hindi, scored in about 22 ms on a single CPU core.
              </p>
              <p className="utterance" lang="hi">
                मुझे कल शाम के लिए <span className="gap"><b>900 ms</b></span> दो टिकट चाहिए।
              </p>
              <span className="more">Open the detector →</span>
            </Link>

            <div className="product card" id="arena">
              <span className="tag">In development</span>
              <h3>Stress-test arena</h3>
              <p>
                A scripted caller that holds real-time spoken conversations with a voice agent and sets
                the traps real callers do: pausing mid-thought, interrupting, changing their mind,
                reading out numbers over street noise. In Hindi, Hinglish and Indian English.
              </p>
              <ul className="traps">
                <li>Mid-thought pauses</li>
                <li>Barge-in</li>
                <li>“Haan”, “hmm” while the agent talks</li>
                <li>Numbers and addresses</li>
                <li>Background noise</li>
                <li>Change of mind</li>
              </ul>
              <span className="more muted">No results yet. The leaderboard will list only runs that happened.</span>
            </div>
          </div>
        </section>

        <section className="section">
          <p className="eyebrow">How the work is done</p>
          <h2>Numbers you can check<span className="danda" aria-hidden="true">।</span></h2>
          <div className="principles">
            <div>
              <h3>Tested the way it is used</h3>
              <p>
                The first model scored 95% on clips and still cut off half of all speakers. So the
                headline test replays whole, unedited recordings and counts interruptions.
              </p>
            </div>
            <div>
              <h3>Losses are published</h3>
              <p>
                On another team&apos;s test set, their model beats ours by 17 to 19 points. That table is
                on the product page, next to the one we win.
              </p>
            </div>
            <div>
              <h3>One command rebuilds it</h3>
              <p>
                Data, training, evaluation and export are a versioned pipeline. Every figure on this
                site traces to a results file and the commit that produced it.
              </p>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
