import type { Metadata } from "next";
import { Demo } from "@/components/Demo";
import { Footer, Nav } from "@/components/Nav";
import { TradeoffChart } from "@/components/TradeoffChart";
import { awayGame, policies, site } from "@/lib/site";

export const metadata: Metadata = {
  title: "End-of-turn detector",
  description: "A small model that tells a voice agent whether the speaker has finished or is only pausing.",
};

const curl = `curl -X POST ${site.api}/predict \\
  -F "file=@speech_so_far.wav"

{
  "probability": 0.094,
  "complete": false,
  "seconds_received": 5.63,
  "inference_ms": 226.2
}`;

const python = `from indic_eot import EndOfTurnDetector

detector = EndOfTurnDetector.load()

# audio: mono float32 at 16 kHz
p = detector.predict(audio)
if p >= 0.5:
    reply()            # the speaker is done
else:
    keep_listening()   # only a pause`;

export default function Viraam() {
  return (
    <>
      <section className="hero hero-product">
        <Nav />
        <div className="hero-body">
          <p className="eyebrow">End-of-turn detector</p>
          <h1>Finished, or just pausing?</h1>
          <p className="lede">
            A fixed silence timer cannot tell. Viraam listens to how the last eight seconds of speech
            sound and answers at every pause, so the agent replies fast when you are done and waits
            when you are not.
          </p>
          <dl className="facts">
            <div><dd>22 ms</dd><dt>to score a pause, one CPU core</dt></div>
            <div><dd>10 MB</dd><dt>model, runs without a GPU</dt></div>
            <div><dd>200 ms</dd><dt>of silence before the first answer</dt></div>
          </dl>
        </div>
      </section>

      <main>
        <section className="section section-night" id="demo">
          <p className="eyebrow">Live</p>
          <h2>Speak, pause, carry on<span className="danda" aria-hidden="true">।</span></h2>
          <p className="section-lede">
            Say a sentence and stop in the middle to think. Your voice rises as a ribbon of lines. At each pause
            the model leaves a tag: amber to keep waiting, green to reply.
          </p>
          <Demo />
          <p className="fine">
            The model runs inside this browser tab, so your audio never leaves your device. Scoring is slower
            here than the 22 ms measured on a server CPU. Sample recordings are from the FLEURS corpus (CC BY 4.0).
          </p>
        </section>

        <section className="section">
          <p className="eyebrow">What happens at a pause</p>
          <div className="steps">
            <div className="card"><span>1</span><h3>You go quiet</h3><p>After 200 ms of silence the agent asks the model. A timer would still be counting.</p></div>
            <div className="card"><span>2</span><h3>The model listens back</h3><p>It hears the last 8 seconds: the falling pitch and fading energy of an ending, or the held tone of a thought in progress.</p></div>
            <div className="card"><span>3</span><h3>The agent replies or waits</h3><p>Confident you finished: reply now. Otherwise keep listening, for up to 1.5 seconds.</p></div>
          </div>
        </section>

        <section className="section" id="results">
          <p className="eyebrow">Results</p>
          <h2>Measured, including where it loses<span className="danda" aria-hidden="true">।</span></h2>
          <div className="results">
            <div className="card">
              <h3>As an agent&apos;s turn-taking rule</h3>
              <p className="cap">406 unedited Hindi test recordings. “Cut off” means the agent replied before the speaker had finished.</p>
              <TradeoffChart />
              <div className="table-scroll">
                <table>
                  <thead><tr><th>Rule</th><th>Speakers cut off</th><th>Mean wait</th></tr></thead>
                  <tbody>
                    {policies.map((p) => (
                      <tr key={p.name} className={p.ours ? "ours" : ""}>
                        <td>{p.name}</td><td>{p.cut}%</td><td>{p.wait.toLocaleString("en")} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="takeaway">Faster and less interrupting than a 500 ms timer. It does not yet reach the low cut-off rate of a 700 ms timer; pauses at commas are the main cause.</p>
            </div>
            <div className="card">
              <h3>Against Smart Turn v3.2, on its own test set</h3>
              <p className="cap">Pipecat&apos;s published test clips (synthetic speech). Smart Turn trained on this collection; Viraam had never seen it. Accuracy:</p>
              <div className="bars">
                {awayGame.map((row) => (
                  <div key={row.language}>
                    <p>{row.language} <span>{row.clips} clips</span></p>
                    <div className="bar ours-bar" style={{ width: `${row.ours}%` }}><b>Viraam {row.ours}%</b></div>
                    <div className="bar" style={{ width: `${row.smartTurn}%` }}><b>Smart Turn {row.smartTurn}%</b></div>
                  </div>
                ))}
              </div>
              <p className="takeaway">Smart Turn wins clearly here. On Viraam&apos;s own test recordings the order reverses: it recognises 98.3% of true endings against Smart Turn&apos;s 75.9%. Each model is strongest on speech like its training data. Widening Viraam&apos;s training data is the next piece of work.</p>
            </div>
          </div>
        </section>

        <section className="section" id="api">
          <p className="eyebrow">API</p>
          <h2>One request per pause<span className="danda" aria-hidden="true">।</span></h2>
          <p className="section-lede">Send the speech so far, up to the pause. You get back the probability that the turn is complete.</p>
          <div className="code">
            <figure><figcaption>HTTP</figcaption><pre>{curl}</pre></figure>
            <figure><figcaption>Python, no network</figcaption><pre>{python}</pre></figure>
          </div>
          <p className="fine">
            The HTTP API is open: no key, no sign-up. Send a WAV file, or raw 32-bit float samples at 16 kHz as{" "}
            <code>application/octet-stream</code>. Requests are limited to 4 MB and only the last 8 seconds are
            scored. The hosted API takes around 200 ms per request and longer on the first call after a quiet
            spell. The Python package runs the same model in about 22 ms on a server CPU.
          </p>
        </section>

        <section className="section">
          <p className="eyebrow">Limits</p>
          <h2>What it cannot do yet<span className="danda" aria-hidden="true">।</span></h2>
          <ul className="limits">
            <li>It was trained on read Hindi speech from one corpus. Expect weaker results on casual conversation, other accents and other languages.</li>
            <li>It judges sound only, not words. A pause after a clause that could stand alone is often called finished.</li>
            <li>It has not been tested on recordings of people hesitating in real calls.</li>
          </ul>
        </section>
      </main>
      <Footer />
    </>
  );
}
