import { useEffect, useMemo, useState } from "react";
import { Consumer, ModelMeta, Tier, featureRows, tierOf, topExplanations, fmt } from "./lib/explain";
import { featureVector } from "./lib/features";
import { scoreMatrix } from "./lib/onnx";
import { Sparkline } from "./components/Sparkline";
import { Gauge } from "./components/Gauge";
import samplePack from "./data/sample_pack.json";

/** JS getDay of day 0 of the recent 90-day window (2016-08-02, Tuesday). */
const WINDOW_DOW0 = 2;

function App(): React.ReactElement {
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [consumers, setConsumers] = useState<Consumer[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [mode, setMode] = useState<"demo" | "analyze">("demo");
  const [paste, setPaste] = useState("");
  const [pasteResult, setPasteResult] = useState<null | {
    score: number;
    tier: Tier;
    features: number[];
  }>(null);

  useEffect(() => {
    (async () => {
      try {
        const m: ModelMeta = await (await fetch("/model/model_meta.json")).json();
        setMeta(m);
        const pack = samplePack as Consumer[];
        const feats = pack.map((c) => featureVector(c.series, WINDOW_DOW0));
        const scores = await scoreMatrix(feats);
        setConsumers(pack.map((c, i) => ({ ...c, score: scores[i] })));
        setSel(pack[0]?.id ?? null);
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    })();
  }, []);

  const ranked = useMemo(
    () => [...consumers].sort((a, b) => b.score - a.score),
    [consumers]
  );
  const selected = ranked.find((c) => c.id === sel) ?? ranked[0];
  const selFeatures = useMemo(
    () => (selected ? featureVector(selected.series, WINDOW_DOW0) : null),
    [selected]
  );

  async function analyze(): Promise<void> {
    if (!meta) return;
    const nums: (number | null)[] = paste
      .split(/[\s,;]+/)
      .filter(Boolean)
      .map((t) => (t === "NA" || t === "?" ? null : Number(t)))
      .map((v) => (isFinite(v as number) ? v : null));
    if (nums.length < 30) {
      setErr("Paste at least 30 daily readings (space/comma/newline separated; NA for missing).");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const feats = featureVector(nums, WINDOW_DOW0);
      const [score] = await scoreMatrix([feats]);
      setPasteResult({ score, tier: tierOf(score, meta.thresholds), features: feats });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (err && !consumers.length) {
    return (
      <div className="wrap">
        <h1>⚡ GridGuard</h1>
        <p className="muted">Error loading model: {err}</p>
      </div>
    );
  }

  return (
    <div className="wrap">
      <header>
        <div>
          <h1>⚡ GridGuard</h1>
          <p className="tagline">
            Electricity-theft detection from smart-meter data · scoring runs entirely in your browser
          </p>
        </div>
        <div className="hdr-metrics">
          {meta && (
            <>
              <div>
                <span className="mnum">{String(meta.metrics.consumer_mean.pr_auc)}</span>
                <span className="mlab">PR-AUC (consumer)</span>
              </div>
              <div>
                <span className="mnum">{String(meta.metrics.consumer_mean.roc_auc)}</span>
                <span className="mlab">ROC-AUC (consumer)</span>
              </div>
              <div>
                <span className="mnum">{String(meta.metrics.consumer_mean.lift_top1p)}x</span>
                <span className="mlab">lift @ top 1%</span>
              </div>
            </>
          )}
        </div>
      </header>

      <nav className="tabs">
        <button className={mode === "demo" ? "on" : ""} onClick={() => setMode("demo")}>
          Watchlist demo
        </button>
        <button className={mode === "analyze" ? "on" : ""} onClick={() => setMode("analyze")}>
          Analyze your own series
        </button>
      </nav>

      {mode === "demo" ? (
        <main className="grid">
          <section className="list">
            <div className="list-head">
              <span>Consumer · 90-day profile</span>
              <span>Risk score</span>
            </div>
            {busy && !consumers.length && (
              <p className="muted pad">Scoring 100 consumers in-browser…</p>
            )}
            {ranked.map((c) => {
              const t = meta ? tierOf(c.score, meta.thresholds) : "ok";
              return (
                <button
                  key={c.id}
                  className={`row ${sel === c.id ? "sel" : ""}`}
                  onClick={() => setSel(c.id)}
                >
                  <span className="cid">{c.id}</span>
                  <Sparkline series={c.series} highlight={t === "critical"} />
                  <span className={`score t-${t}`}>{(c.score * 100).toFixed(1)}%</span>
                </button>
              );
            })}
          </section>
          <aside className="detail">
            {selected && meta && selFeatures && (() => {
              const tier = tierOf(selected.score, meta.thresholds);
              return (
                <>
                  <div className="d-head">
                    <div>
                      <h2>{selected.id}</h2>
                      <p className="muted">
                        90-day window · SGCC ground truth:{" "}
                        <b className={selected.flag ? "flag-theft" : "flag-ok"}>
                          {selected.flag ? "THEFT (flagged)" : "honest"}
                        </b>
                      </p>
                    </div>
                    <Gauge score={selected.score} thresholds={meta.thresholds} tier={tier} />
                  </div>
                  <div className="tier-note" style={{ borderLeftColor: `var(--tier-${tier})` }}>
                    {TIER_NOTE(tier)}
                  </div>
                  <SparklineBig series={selected.series} />
                  <h3>Why flagged?</h3>
                  <WhyList features={selFeatures} meta={meta} />
                  <h3>Feature deviations <span className="muted small">(vs honest Q1–Q3 bands)</span></h3>
                  <div className="tbl-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th><th>Value</th><th>Honest Q1–Q3</th><th>Dev</th><th>Imp</th>
                        </tr>
                      </thead>
                      <tbody>
                        {featureRows(selFeatures, meta, meta.feature_names)
                          .sort((a, b) => Math.abs(b.deviation) * (0.2 + b.importance) - Math.abs(a.deviation) * (0.2 + a.importance))
                          .map((r) => (
                            <tr key={r.name} className={r.outsideBand ? "out" : ""}>
                              <td>{r.name}</td>
                              <td>{fmt(r.value)}</td>
                              <td className="muted">{fmt(r.q25, 2)} – {fmt(r.q75, 2)}</td>
                              <td className={r.deviation > 0 ? "pos" : "neg"}>
                                {r.deviation > 0 ? "+" : ""}{fmt(r.deviation, 1)}
                              </td>
                              <td className="muted">{(r.importance * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </>
              );
            })()}
          </aside>
        </main>
      ) : (
        <main className="analyze">
          <p className="muted">
            Paste up to 90 days of daily kWh readings (most recent last). Missing day = <code>NA</code>.
            The same 22-feature engine + ONNX model run locally — nothing leaves this page.
          </p>
          <textarea
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            placeholder={"12.4 13.1 NA 11.9 0 0 0 …"}
            rows={6}
          />
          <button className="cta" onClick={analyze} disabled={busy}>
            {busy ? "Scoring…" : "Score series →"}
          </button>
          {pasteResult && meta && (
            <div className="paste-res">
              <Gauge score={pasteResult.score} thresholds={meta.thresholds} tier={pasteResult.tier} />
              <div>
                <h2>{(pasteResult.score * 100).toFixed(1)}% theft probability</h2>
                <p className="muted">{TIER_NOTE(pasteResult.tier)}</p>
                <WhyList features={pasteResult.features} meta={meta} />
              </div>
            </div>
          )}
          {err && mode === "analyze" && <p className="err">{err}</p>}
        </main>
      )}

      <footer>
        <p className="muted small">
          Trained on the public SGCC benchmark (42,372 consumers × 1,034 days · 8.5% flagged theft).
          Decision-support only — never a sole basis for enforcement. See ETHICS.md.
        </p>
      </footer>
    </div>
  );
}

function TIER_NOTE(t: Tier): string {
  const map: Record<Tier, string> = {
    critical: "High-confidence anomaly — prioritize for field inspection. Human review required before any action.",
    high: "Anomalous pattern detected — queue for secondary review.",
    watch: "Mild deviation from honest bands — re-score after the next reading window.",
    ok: "Within honest-consumer behavior bands — no action needed.",
  };
  return map[t];
}

function WhyList({ features, meta }: { features: number[]; meta: ModelMeta }): React.ReactElement {
  const why = topExplanations(featureRows(features, meta, meta.feature_names));
  if (!why.length) return <ul className="why"><li className="muted">All features within honest bands.</li></ul>;
  return (
    <ul className="why">
      {why.map((w, i) => <li key={i}>{w.text}</li>)}
    </ul>
  );
}

function SparklineBig({ series }: { series: (number | null)[] }): React.ReactElement {
  const pts = series.map((v, i) => ({ v, i })).filter((p) => p.v !== null) as { v: number; i: number }[];
  const max = Math.max(...pts.map((p) => p.v), 1e-9);
  const W = 560, H = 120;
  const x = (i: number) => (i / Math.max(1, series.length - 1)) * (W - 2) + 1;
  const y = (v: number) => H - 4 - (v / max) * (H - 10);
  const d = pts.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  return (
    <svg className="bigspark" width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <path d={`${d} L${W - 1},${H} L1,${H} Z`} fill="#38bdf822" />
      <path d={d} fill="none" stroke="#38bdf8" strokeWidth="1.6" />
    </svg>
  );
}

export default App;
