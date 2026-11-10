import { lazy, Suspense, useState } from 'react'
import {
  PAPER, LINKS, VIDEO, QUICKSTART, ABSTRACT, HIGHLIGHTS, BENCHMARK, PREFERENCE, EXPERT,
  BIBTEX,
} from './content'

// three.js + drei is roughly 300 kB gzipped. Splitting it out lets the title,
// abstract and results render immediately instead of waiting on WebGL code.
const ComparisonViewer = lazy(() => import('./ComparisonViewer'))

function ViewerFallback() {
  return (
    <section id="demo" className="section viewer-section">
      <div className="section-inner">
        <h2 className="section-title">Interactive comparison</h2>
        <div className="viewer-placeholder">Loading the 3D viewer…</div>
      </div>
    </section>
  )
}

function Hero() {
  const superscripts = PAPER.affiliations.length > 1
  return (
    <header className="hero">
      <div className="section-inner">
        <p className="venue-badge">{PAPER.venue} · {PAPER.session}</p>
        <h1 className="hero-title">{PAPER.title}</h1>

        <p className="authors">
          {PAPER.authors.map((a, i) => (
            <span key={a.name}>
              {a.name}
              {superscripts && <sup>{a.affiliation}</sup>}
              {i < PAPER.authors.length - 1 && <span className="sep">·</span>}
            </span>
          ))}
        </p>
        <p className="affiliations">
          {PAPER.affiliations.map((aff, i) => (
            <span key={aff}>
              {superscripts && <sup>{i + 1}</sup>}
              {aff}
            </span>
          ))}
        </p>

        <nav className="link-row" aria-label="Paper resources">
          {LINKS.paper && (
            <a className="link-btn link-btn-primary" href={LINKS.paper}>
              <span aria-hidden="true">📄</span> Paper
            </a>
          )}
          {LINKS.video && (
            <a className="link-btn" href={LINKS.video} target="_blank" rel="noreferrer">
              <span aria-hidden="true">▶</span> Video
            </a>
          )}
          {LINKS.code && (
            <a className="link-btn" href={LINKS.code} target="_blank" rel="noreferrer">
              <span aria-hidden="true">⌨</span> Code
            </a>
          )}
          {LINKS.ismir && (
            <a className="link-btn" href={LINKS.ismir} target="_blank" rel="noreferrer">
              <span aria-hidden="true">◎</span> ISMIR page
            </a>
          )}
          <a className="link-btn" href="#demo">
            <span aria-hidden="true">◇</span> 3D demo
          </a>
          <a className="link-btn" href={LINKS.bibtexAnchor}>
            <span aria-hidden="true">❝</span> BibTeX
          </a>
        </nav>

        <p className="tagline">{PAPER.tagline}</p>
      </div>
    </header>
  )
}

function Abstract() {
  return (
    <section className="section">
      <div className="section-inner section-narrow">
        <h2 className="section-title">Abstract</h2>
        <p className="abstract">{ABSTRACT}</p>
      </div>
    </section>
  )
}

function Video() {
  return (
    <section id="video" className="section">
      <div className="section-inner section-narrow">
        <h2 className="section-title">Video</h2>
        <div className="video-frame">
          <iframe
            src={`https://www.youtube-nocookie.com/embed/${VIDEO.id}?rel=0`}
            title={VIDEO.title}
            loading="lazy"
            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        </div>
      </div>
    </section>
  )
}

function Highlights() {
  return (
    <section className="section section-alt">
      <div className="section-inner">
        <h2 className="section-title">What the paper shows</h2>
        <div className="card-grid">
          {HIGHLIGHTS.map((h, i) => (
            <article className="card" key={h.heading}>
              <span className="card-index">{String(i + 1).padStart(2, '0')}</span>
              <h3 className="card-heading">{h.heading}</h3>
              <p className="card-body">{h.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function Method() {
  return (
    <section id="method" className="section">
      <div className="section-inner">
        <h2 className="section-title">Method</h2>
        <p className="section-lead">
          A four-stage hybrid pipeline. Stage 2 uses a diffusion model because torso sway
          is genuinely one-to-many. Stages 1, 3 and 4 use deterministic regression because
          their targets are unimodal once conditioned. Stage boundaries and the shoulder
          grouping both come from the coupling analysis, not from a fixed kinematic tree.
        </p>
        <figure className="figure">
          <a href="assets/pipeline.svg" target="_blank" rel="noreferrer" title="Open full resolution">
            <img
              src="assets/pipeline.png"
              alt="Pianoid pipeline. MIDI feeds Stage 1 (wrists, BiLSTM), which feeds Stage 2 (10-joint backbone of head, torso and shoulders, diffusion), which feeds Stage 3 (elbows, BiLSTM) and Stage 4 (lower body, FiLM on spine)."
              loading="lazy"
            />
          </a>
          <figcaption>
            The Pianoid pipeline. Diffusion is reserved for the one-to-many backbone
            (Stage 2). Shoulders are grouped with the backbone rather than the arms,
            following the coupling analysis. Click to open at full resolution.
          </figcaption>
        </figure>
      </div>
    </section>
  )
}

function Results() {
  return (
    <section id="results" className="section section-alt">
      <div className="section-inner">
        <h2 className="section-title">Results</h2>

        <h3 className="sub-title">Benchmark, 80 test sequences</h3>
        <p className="section-lead">
          Pianoid has the highest MPJPE of the four systems and the best Δ₁. That is the
          point. The baselines buy joint-position error by collapsing toward the mean pose,
          which the expressiveness ratios expose.
        </p>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>{BENCHMARK.columns.map(c => <th key={c} scope="col">{c}</th>)}</tr>
            </thead>
            <tbody>
              {BENCHMARK.rows.map(r => (
                <tr key={r.label} className={r.ours ? 'row-ours' : r.gt ? 'row-gt' : undefined}>
                  <th scope="row">
                    {r.label}
                    {r.note && <span className="marker">{r.note}</span>}
                  </th>
                  {r.cells.map((c, i) => (
                    <td
                      key={i}
                      className={
                        r.best.includes(i) ? 'cell-best'
                          : r.second.includes(i) ? 'cell-second'
                            : undefined
                      }
                    >
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="footnote">{BENCHMARK.footnote}</p>

        <h3 className="sub-title">Non-expert preference, N = 28</h3>
        <p className="section-lead">{PREFERENCE.caption}</p>

        <p className="bar-group-label">Share of trials preferring Pianoid</p>
        <ul className="bars">
          {PREFERENCE.rows.map(r => (
            <li className="bar-row" key={r.label}>
              <span className="bar-label">{r.label}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${r.value}%` }} />
              </span>
              <span className="bar-value">{r.value.toFixed(1)}%</span>
            </li>
          ))}
        </ul>

        <p className="bar-group-label">Reverse direction, for reference</p>
        <ul className="bars">
          <li className="bar-row">
            <span className="bar-label">{PREFERENCE.gt.label}</span>
            <span className="bar-track">
              <span
                className="bar-fill bar-fill-gt"
                style={{ width: `${PREFERENCE.gt.value}%` }}
              />
            </span>
            <span className="bar-value">{PREFERENCE.gt.value.toFixed(1)}%</span>
          </li>
        </ul>

        <div className="bar-scale" aria-hidden="true">
          <span className="bar-scale-ticks">
            <span>0%</span>
            <span>50% (chance)</span>
            <span>100%</span>
          </span>
        </div>

        <p className="footnote">{PREFERENCE.gt.note}</p>

        <h3 className="sub-title">Expert diagnosis, N = 9</h3>
        <p className="section-lead">{EXPERT.caption}</p>
        <ul className="theme-list">
          {EXPERT.themes.map(t => (
            <li className="theme" key={t.id}>
              <span className="theme-id">{t.id}</span>
              <span className="theme-agree">{t.agreement}</span>
              <span className="theme-text">{t.text}</span>
            </li>
          ))}
        </ul>
        <p className="takeaway">{EXPERT.takeaway}</p>
      </div>
    </section>
  )
}

function Code() {
  return (
    <section id="code" className="section section-alt">
      <div className="section-inner section-narrow">
        <h2 className="section-title">Code and checkpoints</h2>
        <p className="section-lead">
          The <a href={LINKS.code} target="_blank" rel="noreferrer">repository</a> holds the
          full pipeline, every Table 2 baseline adaptation and the Table 3 ablations.
          The four released stage <a href={LINKS.checkpoints} target="_blank" rel="noreferrer">checkpoints</a> reproduce
          the headline numbers without training. Motion data comes from
          the <a href={LINKS.data} target="_blank" rel="noreferrer">MOSA corpus</a> (CC BY 4.0),
          which is downloaded separately.
        </p>
        <pre className="bibtex"><code>{QUICKSTART}</code></pre>
      </div>
    </section>
  )
}

function BibTeX() {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }
  return (
    <section id="bibtex" className="section">
      <div className="section-inner section-narrow">
        <div className="bibtex-head">
          <h2 className="section-title">BibTeX</h2>
          <button className="btn-copy" onClick={copy}>
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        </div>
        <pre className="bibtex"><code>{BIBTEX}</code></pre>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <div className="section-inner section-narrow">
        <p>
          {PAPER.title}. {PAPER.venueLong}.
        </p>
        <p className="footer-dim">
          Ground-truth motion is from the public MOSA corpus. Both user studies were run
          with informed consent under institutional review board approval. The model is not
          identity-conditioned, and identity-conditioned extensions are deliberately not
          pursued.
        </p>
      </div>
    </footer>
  )
}

export default function ShowcasePage() {
  return (
    <div className="page">
      <a className="skip-link" href="#demo">Skip to the interactive demo</a>
      <Hero />
      <Abstract />
      <Video />
      <Highlights />
      <Method />
      <Suspense fallback={<ViewerFallback />}>
        <ComparisonViewer />
      </Suspense>
      <Results />
      <Code />
      <BibTeX />
      <Footer />
    </div>
  )
}
