import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Line, View, PerspectiveCamera } from '@react-three/drei'
import { TextureLoader, DoubleSide, type PerspectiveCamera as ThreePerspectiveCamera } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { VARIANTS, CURATED_IDX, type Variant } from './content'

// ─── 26-joint skeleton constants ───────────────────────────
const EDGES: [number, number][] = [
  [0,1],[2,3],[0,2],[1,3],
  [2,4],[3,4],
  [4,6],
  [5,7],
  [4,8],[4,12],[5,8],[5,12],
  [8,9],[9,10],[10,11],
  [12,13],[13,14],[14,15],
  [6,18],[6,19],[7,16],[7,17],
  [16,17],[18,19],[16,18],[17,19],
  [16,20],[18,20],[20,21],[21,22],
  [17,23],[19,23],[23,24],[24,25],
]
const SKIP_JOINTS = new Set([11, 15])
const ACTIVE_EDGES = EDGES.filter(([a, b]) => !SKIP_JOINTS.has(a) && !SKIP_JOINTS.has(b))

type SampleInfo = {
  idx: number
  composer: string
  piece: string
  performer: string
  trial: string
  henle: number
  num_frames: number
  fps: number
  mpjpe: number
  has_audio: boolean
  trimmed_frames?: number
}

type KeyboardData = {
  center_x: number
  front_y: number
  back_y: number
  z: number
  width: number
  depth: number
}

type FullInfo = SampleInfo & { keyboard?: KeyboardData }

// Tuned against the light viewport surface (--panel-bg #edeff3), not a dark one.
const GT_JOINT  = '#39414f'
const GT_EDGE   = '#6b7688'
const VAR_JOINT = '#1a3fb8'
const VAR_EDGE  = '#2563eb'
const GRID_MAIN = '#c2c9d4'
const GRID_LINE = '#dce0e7'

function Skeleton({
  positions, jointColor, edgeColor,
}: {
  positions: Float32Array | null
  jointColor: string
  edgeColor: string
}) {
  const joints = useMemo(() => {
    if (!positions) return []
    const pts: [number, number, number][] = []
    for (let i = 0; i < 26; i++) {
      // MOSA (X=lat, Y=depth, Z=height) → three.js (X=−lat, Y=Z, Z=Y)
      pts.push([-positions[i*3], positions[i*3+2], positions[i*3+1]])
    }
    return pts
  }, [positions])

  if (!positions) return null
  return (
    <group>
      {joints.map((p, i) => {
        if (SKIP_JOINTS.has(i)) return null
        return (
          <mesh key={i} position={p}>
            <sphereGeometry args={[8, 8, 8]} />
            <meshStandardMaterial color={jointColor} />
          </mesh>
        )
      })}
      {ACTIVE_EDGES.map(([a, b], i) => {
        const pa = joints[a], pb = joints[b]
        if (!pa || !pb) return null
        return <Line key={i} points={[pa, pb]} color={edgeColor} lineWidth={1.5} />
      })}
    </group>
  )
}

function KeyboardPlane({ kbd }: { kbd: KeyboardData | null }) {
  const texture = useMemo(
    () => new TextureLoader().load(import.meta.env.BASE_URL + 'assets/keyboard.png'),
    [],
  )
  if (!kbd) return null
  const cy_mid = (kbd.front_y + kbd.back_y) / 2
  return (
    <mesh
      position={[-kbd.center_x, kbd.z, cy_mid]}
      rotation={[-Math.PI / 2, 0, Math.PI]}
    >
      <planeGeometry args={[kbd.width, kbd.depth]} />
      <meshStandardMaterial map={texture} transparent opacity={0.85} side={DoubleSide} />
    </mesh>
  )
}

function PanelContent({
  positions, kbd, controlsRef, onControlsChange, jointColor, edgeColor,
}: {
  positions: Float32Array | null
  kbd: KeyboardData | null
  controlsRef: React.RefObject<OrbitControlsImpl | null>
  onControlsChange: () => void
  jointColor: string
  edgeColor: string
}) {
  return (
    <>
      <PerspectiveCamera makeDefault position={[-1350, 400, 600]} fov={50} near={1} far={10000} />
      <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.1}
                     target={[0, 0, -400]} onChange={onControlsChange} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[500, 1000, 500]} intensity={0.6} />
      <gridHelper args={[2000, 20, GRID_MAIN, GRID_LINE]} />
      <KeyboardPlane kbd={kbd} />
      <Skeleton positions={positions} jointColor={jointColor} edgeColor={edgeColor} />
    </>
  )
}

export default function ComparisonViewer() {
  const [manifest, setManifest] = useState<SampleInfo[]>([])
  const [selectedIdx, setSelectedIdx] = useState<number>(CURATED_IDX[0])
  const [variant, setVariant] = useState<Variant>(VARIANTS[0])

  const [info, setInfo] = useState<FullInfo | null>(null)
  const [dataGT, setDataGT]   = useState<Float32Array | null>(null)
  const [dataVar, setDataVar] = useState<Float32Array | null>(null)
  const [variantStatus, setVariantStatus] = useState<'ok' | 'missing' | 'loading'>('loading')

  const [maxFrame, setMaxFrame] = useState(0)
  const [currentTimeMs, setCurrentTimeMs] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastTickRef = useRef<number | null>(null)

  // Camera sync between the two <View> panels
  const controlsARef = useRef<OrbitControlsImpl | null>(null)
  const controlsBRef = useRef<OrbitControlsImpl | null>(null)
  const syncingRef = useRef(false)
  const syncControls = useCallback((source: 'A' | 'B') => {
    if (syncingRef.current) return
    const src = (source === 'A' ? controlsARef : controlsBRef).current
    const dst = (source === 'A' ? controlsBRef : controlsARef).current
    if (!src || !dst) return
    // Both panels are driven by <PerspectiveCamera>, so this narrowing holds.
    const srcCam = src.object as ThreePerspectiveCamera
    const dstCam = dst.object as ThreePerspectiveCamera
    syncingRef.current = true
    try {
      dstCam.position.copy(srcCam.position)
      dst.target.copy(src.target)
      dstCam.zoom = srcCam.zoom
      dstCam.updateProjectionMatrix()
      dst.update()
    } finally {
      syncingRef.current = false
    }
  }, [])
  const onChangeA = useCallback(() => syncControls('A'), [syncControls])
  const onChangeB = useCallback(() => syncControls('B'), [syncControls])

  const viewARef = useRef<HTMLDivElement>(null!)
  const viewBRef = useRef<HTMLDivElement>(null!)
  const canvasContainerRef = useRef<HTMLDivElement>(null!)
  const sectionRef = useRef<HTMLElement>(null!)
  const FPS = 60

  // Stop the WebGL loop while the viewer is scrolled out of sight. The canvas
  // is viewport-fixed (drei's <View> pattern), so without this it keeps
  // rendering while the reader is on the abstract or the BibTeX block.
  const [inView, setInView] = useState(true)
  useEffect(() => {
    const el = sectionRef.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { rootMargin: '200px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  // Derived current frame
  const totalMs = (maxFrame / FPS) * 1000
  const frame = totalMs > 0
    ? Math.min(maxFrame - 1, Math.floor((currentTimeMs / 1000) * FPS))
    : 0

  // ── Load manifest ──
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'data/manifest.json')
      .then(r => r.json())
      .then((m: SampleInfo[]) => setManifest(m))
      .catch(console.error)
  }, [])

  // ── Load clip data when piece or variant changes ──
  useEffect(() => {
    if (!manifest.length) return
    const sample = manifest.find(s => s.idx === selectedIdx)
    if (!sample) return

    const dir = `${import.meta.env.BASE_URL}data/${String(selectedIdx).padStart(5, '0')}`
    setDataGT(null)
    setDataVar(null)
    setVariantStatus('loading')
    setCurrentTimeMs(0)
    setIsPlaying(false)
    lastTickRef.current = null

    // GT and info always available
    Promise.all([
      fetch(`${dir}/gt.bin`).then(r => r.arrayBuffer()),
      fetch(`${dir}/info.json`).then(r => r.json()).catch(() => null),
    ]).then(([bufGT, infoJson]) => {
      const gt = new Float32Array(bufGT as ArrayBuffer)
      setDataGT(gt)
      setMaxFrame(gt.length / (26 * 3))
      setInfo({ ...sample, keyboard: infoJson?.keyboard })
    }).catch(console.error)

    // Variant — may be missing on disk. Vite dev server serves index.html as
    // a 200 SPA fallback for unknown paths, so we also reject on text/html.
    fetch(`${dir}/${variant.file}`)
      .then(r => {
        if (!r.ok) throw new Error('missing')
        const ct = r.headers.get('content-type') ?? ''
        if (ct.includes('text/html')) throw new Error('spa-fallback')
        return r.arrayBuffer()
      })
      .then(buf => {
        const fa = new Float32Array(buf as ArrayBuffer)
        // Sanity: sometimes the static server may hand back an empty file.
        if (fa.length < 26 * 3) throw new Error('empty')
        setDataVar(fa)
        setVariantStatus('ok')
      })
      .catch(() => {
        setDataVar(null)
        setVariantStatus('missing')
      })
  }, [manifest, selectedIdx, variant])

  // ── Animation loop ──
  useEffect(() => {
    if (!isPlaying || totalMs <= 0 || !dataGT) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTickRef.current = null
      if (audioRef.current && !audioRef.current.paused) audioRef.current.pause()
      return
    }
    const tick = (now: number) => {
      if (lastTickRef.current === null) lastTickRef.current = now
      const dt = (now - lastTickRef.current) * playbackRate
      lastTickRef.current = now
      setCurrentTimeMs(prev => {
        const next = prev + dt
        return next >= totalMs ? 0 : next
      })
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      lastTickRef.current = null
    }
  }, [isPlaying, totalMs, dataGT, playbackRate])

  // Scrolling away should not leave audio playing behind the reader.
  useEffect(() => {
    if (!inView) setIsPlaying(false)
  }, [inView])

  // ── Audio sync ──
  // The deploy bundle has audio.mp3 already trimmed to the motion segment, and
  // every .bin is clipped to the same range — so audio time aligns 1:1 with
  // motion time, no offset needed.
  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const targetTime = currentTimeMs / 1000
    if (isPlaying) {
      if (Math.abs(audio.currentTime - targetTime) > 0.15) {
        audio.currentTime = targetTime
      }
      if (audio.paused) audio.play().catch(() => {/* autoplay gesture */})
      audio.playbackRate = playbackRate
    } else if (!audio.paused) {
      audio.pause()
    }
  }, [isPlaying, currentTimeMs, playbackRate])

  // ── Per-frame slice ──
  const posGT = useMemo(() => {
    if (!dataGT) return null
    const start = frame * 26 * 3
    return dataGT.subarray(start, start + 26 * 3)
  }, [dataGT, frame])

  const posVar = useMemo(() => {
    if (!dataVar) return null
    const start = frame * 26 * 3
    return dataVar.subarray(start, start + 26 * 3)
  }, [dataVar, frame])

  // ── Transport ──
  const togglePlay = useCallback(() => setIsPlaying(p => !p), [])
  const seekToMs = useCallback((ms: number) => {
    setCurrentTimeMs(Math.max(0, Math.min(totalMs - 1, ms)))
    lastTickRef.current = null
  }, [totalMs])

  // Keyboard shortcuts, but only while the viewer is actually on screen —
  // otherwise space-to-scroll breaks for someone reading the rest of the page.
  useEffect(() => {
    if (!inView) return
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.key === ' ') { e.preventDefault(); togglePlay() }
      else if (e.key === 'ArrowLeft')  seekToMs(currentTimeMs - 1000)
      else if (e.key === 'ArrowRight') seekToMs(currentTimeMs + 1000)
      else if (e.key === 'r' || e.key === 'R') { setCurrentTimeMs(0); setIsPlaying(true) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [inView, togglePlay, seekToMs, currentTimeMs])

  const curatedOptions = useMemo(() => {
    if (!manifest.length) return []
    return CURATED_IDX
      .map(i => manifest.find(s => s.idx === i))
      .filter((s): s is SampleInfo => !!s)
  }, [manifest])

  const audioUrl = info?.has_audio
    ? `${import.meta.env.BASE_URL}data/${String(selectedIdx).padStart(5, '0')}/audio.mp3`
    : null

  return (
    <section id="demo" className="section viewer-section" ref={sectionRef}>
      <div className="section-inner">
        <h2 className="section-title">Interactive comparison</h2>
        <p className="section-lead">
          Recorded ground truth beside a generated condition, driven by the same MIDI and
          synced to the recorded audio. Both cameras are locked together, so dragging
          either panel orbits both. These are the twelve clips used in the non-expert
          2AFC study.
        </p>

        <div className="viewer-controls">
          <label className="field">
            <span className="field-label">Clip</span>
            <select
              value={selectedIdx}
              onChange={e => setSelectedIdx(Number(e.target.value))}
              className="select select-wide"
              disabled={!curatedOptions.length}
            >
              {curatedOptions.map(s => (
                <option key={s.idx} value={s.idx}>
                  {s.composer} · {s.piece} · {s.performer}/{s.trial} · Henle {s.henle} · MPJPE {s.mpjpe.toFixed(1)} mm
                </option>
              ))}
            </select>
          </label>

          <div className="field">
            <span className="field-label">Compare against</span>
            <div className="variant-row" role="group" aria-label="Comparison condition">
              {VARIANTS.map(v => {
                const isActive = v.key === variant.key
                return (
                  <button
                    key={v.key}
                    onClick={() => setVariant(v)}
                    title={v.hint}
                    aria-pressed={isActive}
                    className={isActive ? 'chip chip-active' : 'chip'}
                  >
                    {v.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {(variant.hint || variantStatus === 'missing') && (
          <p className="viewer-hint">
            {variant.hint}
            {variant.citation && <span className="cite"> [{variant.citation}]</span>}
            {variantStatus === 'missing' && (
              <span className="warn"> ⚠ {variant.file} not found for this clip</span>
            )}
          </p>
        )}

        <div className="viewer-frame">
          <div className="viewer-topbar">
            <span>
              {info ? `${info.composer} · ${info.piece}` : 'Loading clip…'}
            </span>
            {info && (
              <span className="meta">
                {info.performer}/{info.trial} · Henle {info.henle}
                {/* The .bin is trimmed to the showcase segment, so num_frames
                    from the manifest would overstate the clip length. */}
                {totalMs > 0 && ` · ${(totalMs / 1000).toFixed(1)} s`} · MPJPE{' '}
                {info.mpjpe.toFixed(1)} mm
              </span>
            )}
          </div>

          <div className="panels" ref={canvasContainerRef}>
            <div className="panel-wrap">
              <span className="panel-label panel-label-gt">Ground truth</span>
              <div ref={viewARef} className="panel" />
            </div>
            <div className="panel-wrap">
              <span className="panel-label panel-label-var">
                {variant.label}
                {variantStatus === 'missing' && <span className="warn"> (unavailable)</span>}
              </span>
              <div ref={viewBRef} className="panel" />
            </div>

            {/* One shared <Canvas>, viewport-fixed, with two <View> portals. */}
            <Canvas
              eventSource={canvasContainerRef as React.RefObject<HTMLElement>}
              className="viewer-canvas"
              frameloop={inView ? 'always' : 'never'}
              gl={{ antialias: true }}
            >
              <View track={viewARef as React.RefObject<HTMLElement>}>
                <PanelContent
                  positions={posGT}
                  kbd={info?.keyboard ?? null}
                  controlsRef={controlsARef}
                  onControlsChange={onChangeA}
                  jointColor={GT_JOINT}
                  edgeColor={GT_EDGE}
                />
              </View>
              <View track={viewBRef as React.RefObject<HTMLElement>}>
                <PanelContent
                  positions={posVar}
                  kbd={info?.keyboard ?? null}
                  controlsRef={controlsBRef}
                  onControlsChange={onChangeB}
                  jointColor={VAR_JOINT}
                  edgeColor={VAR_EDGE}
                />
              </View>
            </Canvas>
          </div>

          <div className="transport">
            <button
              onClick={togglePlay}
              className={isPlaying ? 'btn-play btn-pause' : 'btn-play'}
              disabled={!dataGT}
            >
              {isPlaying ? '❚❚ Pause' : '▶ Play'}
            </button>
            <span className="timecode">
              {(currentTimeMs / 1000).toFixed(1)}s / {(totalMs / 1000).toFixed(1)}s
            </span>
            <input
              type="range" min={0} max={Math.max(1, totalMs)} step={50}
              value={currentTimeMs}
              onChange={e => seekToMs(Number(e.target.value))}
              className="scrub"
              aria-label="Seek"
            />
            <select
              value={playbackRate}
              onChange={e => setPlaybackRate(Number(e.target.value))}
              className="select"
              aria-label="Playback rate"
            >
              <option value={0.25}>0.25×</option>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={1.5}>1.5×</option>
              <option value={2}>2×</option>
            </select>
          </div>
        </div>

        <p className="viewer-foot">
          drag = orbit (both panels) · space = play/pause · ←/→ = seek 1 s · R = restart
        </p>

        {audioUrl && (
          <audio ref={audioRef} src={audioUrl} preload="auto" style={{ display: 'none' }} />
        )}
      </div>
    </section>
  )
}
