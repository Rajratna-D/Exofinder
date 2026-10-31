import React, { useState, useEffect, useRef } from 'react'
import { Volume2, VolumeX, Play, Pause, RotateCcw, Download, Radio, Waves, Activity, Sparkles, Sliders } from 'lucide-react'

export default function SonificationPlayer({
  starId = 77202722,
  starName = "TIC 77202722",
  onClose = null,
  embedded = false
}) {
  // Playback & Audio State
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLooping, setIsLooping] = useState(true)
  const [isMuted, setIsMuted] = useState(false)
  const [volume, setVolume] = useState(0.7)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const [baseFreq, setBaseFreq] = useState(440) // 220, 440, 660
  const [contrast, setContrast] = useState(1.5) // 1.0, 1.5, 2.5
  
  // Scrubber state (0.0 to 1.0 progress across transit window)
  const [progress, setProgress] = useState(0.0)
  const [currentHz, setCurrentHz] = useState(440)
  const [currentDipPpm, setCurrentDipPpm] = useState(0)

  // Photometry profile for audio
  const [transitProfile, setTransitProfile] = useState(null)
  const [audioTelemetry, setAudioTelemetry] = useState(null)
  const [loading, setLoading] = useState(true)

  // Web Audio Context refs
  const audioCtxRef = useRef(null)
  const oscRef = useRef(null)
  const oscHarmonicRef = useRef(null)
  const gainNodeRef = useRef(null)
  const pannerRef = useRef(null)
  const animFrameRef = useRef(null)
  const startTimeRef = useRef(0)
  const startProgressRef = useRef(0)
  const canvasRef = useRef(null)

  // Duration in seconds for full sweep across transit window [-0.15, 0.15] phase
  const sweepDuration = 4.5 / playbackSpeed

  // Fetch lightcurve profile for sonification
  useEffect(() => {
    let isMounted = true
    setLoading(true)
    setIsPlaying(false)
    if (audioCtxRef.current && audioCtxRef.current.state === 'running') {
      audioCtxRef.current.suspend()
    }

    async function fetchSonificationData() {
      try {
        const [lcRes, audioMetaRes] = await Promise.all([
          fetch(`/api/lightcurve/${starId}?max_points=800&phase_window=0.15`).then(r => r.json()),
          fetch(`/api/star/${starId}/audio?format=json&base_freq=${baseFreq}&contrast=${contrast}`).then(r => r.json())
        ])

        if (!isMounted) return

        if (lcRes && lcRes.phase_folded && lcRes.phase_folded.binned_flux) {
          const binCenters = lcRes.phase_folded.bin_centers || []
          const binnedFlux = lcRes.phase_folded.binned_flux || []
          setTransitProfile({
            binCenters,
            binnedFlux,
            period: lcRes.period,
            depth: lcRes.depth,
            duration: lcRes.duration
          })
        }
        if (audioMetaRes) {
          setAudioTelemetry(audioMetaRes)
        }
      } catch (err) {
        console.error("Failed to load sonification profile:", err)
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchSonificationData()

    return () => {
      isMounted = false
      stopAudio()
    }
  }, [starId, baseFreq, contrast])

  // Clean stop audio
  const stopAudio = () => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    try {
      if (gainNodeRef.current && audioCtxRef.current) {
        gainNodeRef.current.gain.setValueAtTime(0, audioCtxRef.current.currentTime)
      }
      if (oscRef.current) {
        oscRef.current.stop()
        oscRef.current.disconnect()
        oscRef.current = null
      }
      if (oscHarmonicRef.current) {
        oscHarmonicRef.current.stop()
        oscHarmonicRef.current.disconnect()
        oscHarmonicRef.current = null
      }
    } catch {
      // safe cleanup
    }
    setIsPlaying(false)
  }

  // Calculate instantaneous flux and pitch at progress t in [0, 1]
  const getFluxAtProgress = (p) => {
    if (!transitProfile || !transitProfile.binnedFlux.length) {
      // Gaussian synthetic fall-back
      const phase = (p - 0.5) * 0.3
      const dip = 0.015 * Math.exp(-Math.pow(phase / 0.02, 2))
      return 1.0 - dip
    }
    const { binCenters, binnedFlux } = transitProfile
    const targetPhase = -0.15 + p * 0.30

    // Linear interpolation
    if (targetPhase <= binCenters[0]) return binnedFlux[0]
    if (targetPhase >= binCenters[binCenters.length - 1]) return binnedFlux[binnedFlux.length - 1]

    for (let i = 0; i < binCenters.length - 1; i++) {
      if (targetPhase >= binCenters[i] && targetPhase <= binCenters[i + 1]) {
        const span = binCenters[i + 1] - binCenters[i]
        const frac = span > 0 ? (targetPhase - binCenters[i]) / span : 0
        return binnedFlux[i] * (1 - frac) + binnedFlux[i + 1] * frac
      }
    }
    return 1.0
  }

  // Calculate pitch frequency in Hz from flux
  const calculateFrequency = (fluxVal) => {
    const baseline = 1.0
    const rawDip = Math.max(0, baseline - fluxVal)
    const maxKnownDip = transitProfile?.depth || 0.015
    const normalizedDip = Math.min(1.0, rawDip / Math.max(1e-4, maxKnownDip))
    
    // Scale frequency drop
    const dropFraction = Math.min(0.45, 0.32 * contrast * normalizedDip)
    return baseFreq * (1.0 - dropFraction)
  }

  // Start or resume Web Audio engine
  const startAudio = () => {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) return

    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContextClass()
    } else if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume()
    }

    const ctx = audioCtxRef.current
    const now = ctx.currentTime

    // Master Gain Node with anti-click envelope
    const masterGain = ctx.createGain()
    masterGain.gain.setValueAtTime(0.0001, now)
    const targetVol = isMuted ? 0 : volume
    masterGain.gain.exponentialRampToValueAtTime(Math.max(0.001, targetVol), now + 0.05)
    gainNodeRef.current = masterGain

    // Lowpass filter to ensure warm celestial timbre
    const filter = ctx.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(1400, now)

    // Stereo Panner (Left = Ingress, Center = Transit Midpoint, Right = Egress)
    let pannerNode = null
    if (ctx.createStereoPanner) {
      pannerNode = ctx.createStereoPanner()
      const initialPan = (progress - 0.5) * 1.6 // -0.8 to +0.8
      pannerNode.pan.setValueAtTime(Math.max(-0.95, Math.min(0.95, initialPan)), now)
      pannerRef.current = pannerNode
    }

    // Primary fundamental oscillator (Warm Sine)
    const osc1 = ctx.createOscillator()
    osc1.type = 'sine'
    const initialFlux = getFluxAtProgress(progress)
    const initialFreq = calculateFrequency(initialFlux)
    osc1.frequency.setValueAtTime(initialFreq, now)
    oscRef.current = osc1

    // Secondary subtle overtone oscillator (Adds presence and scientific chime)
    const osc2 = ctx.createOscillator()
    osc2.type = 'triangle'
    osc2.frequency.setValueAtTime(initialFreq * 2.0, now)
    const overtoneGain = ctx.createGain()
    overtoneGain.gain.setValueAtTime(0.12, now)
    oscHarmonicRef.current = osc2

    // Audio Graph Routing
    osc1.connect(filter)
    osc2.connect(overtoneGain)
    overtoneGain.connect(filter)

    if (pannerNode) {
      filter.connect(pannerNode)
      pannerNode.connect(masterGain)
    } else {
      filter.connect(masterGain)
    }
    masterGain.connect(ctx.destination)

    osc1.start(now)
    osc2.start(now)

    startTimeRef.current = performance.now()
    startProgressRef.current = progress
    setIsPlaying(true)

    // Animation and audio modulation loop
    const animateAudio = (time) => {
      const elapsedSeconds = (time - startTimeRef.current) / 1000
      let currentProgress = startProgressRef.current + (elapsedSeconds / sweepDuration)

      if (currentProgress >= 1.0) {
        if (isLooping) {
          startTimeRef.current = time
          startProgressRef.current = 0.0
          currentProgress = 0.0
        } else {
          stopAudio()
          setProgress(1.0)
          return
        }
      }

      setProgress(currentProgress)

      // Compute instantaneous parameters
      const currentFlux = getFluxAtProgress(currentProgress)
      const freq = calculateFrequency(currentFlux)
      setCurrentHz(Math.round(freq * 10) / 10)
      const dipPpm = Math.max(0, Math.round((1.0 - currentFlux) * 1e6))
      setCurrentDipPpm(dipPpm)

      // Modulate audio frequency
      if (ctx && oscRef.current && oscHarmonicRef.current) {
        const audioNow = ctx.currentTime
        oscRef.current.frequency.setTargetAtTime(freq, audioNow, 0.015)
        oscHarmonicRef.current.frequency.setTargetAtTime(freq * 2.0, audioNow, 0.015)

        // Modulate spatial stereo pan
        if (pannerRef.current) {
          const panVal = (currentProgress - 0.5) * 1.6
          pannerRef.current.pan.setTargetAtTime(Math.max(-0.95, Math.min(0.95, panVal)), audioNow, 0.03)
        }
      }

      animFrameRef.current = requestAnimationFrame(animateAudio)
    }

    animFrameRef.current = requestAnimationFrame(animateAudio)
  }

  // Toggle Play / Pause
  const handleTogglePlay = () => {
    if (isPlaying) {
      stopAudio()
    } else {
      if (progress >= 0.99) {
        setProgress(0.0)
      }
      startAudio()
    }
  }

  // Scrubber change
  const handleScrub = (newProgress) => {
    setProgress(newProgress)
    startProgressRef.current = newProgress
    startTimeRef.current = performance.now()
    const f = getFluxAtProgress(newProgress)
    const hz = calculateFrequency(f)
    setCurrentHz(Math.round(hz * 10) / 10)
    setCurrentDipPpm(Math.max(0, Math.round((1.0 - f) * 1e6)))

    if (isPlaying && audioCtxRef.current && oscRef.current) {
      const now = audioCtxRef.current.currentTime
      oscRef.current.frequency.setValueAtTime(hz, now)
      if (oscHarmonicRef.current) {
        oscHarmonicRef.current.frequency.setValueAtTime(hz * 2, now)
      }
      if (pannerRef.current) {
        pannerRef.current.pan.setValueAtTime((newProgress - 0.5) * 1.6, now)
      }
    }
  }

  // Handle Volume change
  const handleVolumeChange = (newVol) => {
    setVolume(newVol)
    setIsMuted(newVol === 0)
    if (gainNodeRef.current && audioCtxRef.current) {
      gainNodeRef.current.gain.setValueAtTime(newVol, audioCtxRef.current.currentTime)
    }
  }

  // Draw Lightcurve Waveform Canvas with Synchronized Playhead
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    ctx.clearRect(0, 0, width, height)

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height)
    bgGrad.addColorStop(0, 'rgba(4, 6, 10, 0.95)')
    bgGrad.addColorStop(1, 'rgba(10, 16, 28, 0.95)')
    ctx.fillStyle = bgGrad
    ctx.fillRect(0, 0, width, height)

    // Grid lines
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let x = 0; x <= width; x += width / 6) {
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
    }
    for (let y = 0; y <= height; y += height / 4) {
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
    }
    ctx.stroke()

    // Draw baseline 1.0 flux reference
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)'
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(0, height * 0.25)
    ctx.lineTo(width, height * 0.25)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw folded transit curve
    const steps = 120
    ctx.strokeStyle = '#00f2fe'
    ctx.lineWidth = 2.5
    ctx.shadowColor = 'rgba(0, 242, 254, 0.5)'
    ctx.shadowBlur = 8
    ctx.beginPath()

    const minFlux = transitProfile?.binnedFlux ? Math.min(...transitProfile.binnedFlux) : 0.985
    const fluxRange = Math.max(0.015, (1.002 - minFlux) * 1.2)

    for (let i = 0; i <= steps; i++) {
      const p = i / steps
      const f = getFluxAtProgress(p)
      const x = p * width
      const y = height * 0.25 + ((1.0 - f) / fluxRange) * (height * 0.65)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.shadowBlur = 0

    // Draw ingress / egress zone shading
    const midX = width * 0.5
    const transitRadius = width * 0.20
    ctx.fillStyle = 'rgba(255, 138, 0, 0.06)'
    ctx.fillRect(midX - transitRadius, 0, transitRadius * 2, height)

    // Draw Synchronized Audio Laser Playhead
    const playheadX = progress * width
    ctx.strokeStyle = isPlaying ? '#ff8a00' : 'rgba(255, 138, 0, 0.5)'
    ctx.lineWidth = 2.5
    ctx.shadowColor = '#ff8a00'
    ctx.shadowBlur = isPlaying ? 12 : 4
    ctx.beginPath()
    ctx.moveTo(playheadX, 0)
    ctx.lineTo(playheadX, height)
    ctx.stroke()

    // Playhead node glow
    const currentF = getFluxAtProgress(progress)
    const playheadY = height * 0.25 + ((1.0 - currentF) / fluxRange) * (height * 0.65)
    ctx.fillStyle = '#ff8a00'
    ctx.beginPath()
    ctx.arc(playheadX, playheadY, 5.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0
  }, [progress, transitProfile, isPlaying])

  // Download Sonification WAV file
  const handleDownloadWav = () => {
    const url = `/api/star/${starId}/audio?format=wav&base_freq=${baseFreq}&duration=5.0&contrast=${contrast}`
    const a = document.createElement('a')
    a.href = url
    a.download = `${starName.replace(/\s+/g, '_')}_transit_audio.wav`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  // Transit Phase Status Label
  const getPhaseStatus = () => {
    if (progress < 0.32) return { text: 'OUT OF TRANSIT (BASELINE)', color: 'text-cyan-400' }
    if (progress < 0.42) return { text: 'INGRESS OCCURRING', color: 'text-solar-amber' }
    if (progress < 0.58) return { text: 'MID-TRANSIT ECLIPSE CORE', color: 'text-rose-400' }
    if (progress < 0.68) return { text: 'EGRESS TRANSIT DURATION', color: 'text-solar-amber' }
    return { text: 'OUT OF TRANSIT (BASELINE)', color: 'text-cyan-400' }
  }

  const phaseStatus = getPhaseStatus()
  const pitchDrop = Math.max(0, baseFreq - currentHz)

  return (
    <div className={`p-5 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.4)] flex flex-col gap-4 ${embedded ? 'w-full' : 'max-w-2xl mx-auto'}`}>
      {/* ─── Header & Audio Status ─── */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-[#8CE8B5] shadow-[0_0_12px_rgba(140,232,181,0.2)]">
            <Waves className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-mono text-sm font-bold text-slate-100 uppercase tracking-wider">
                Acoustic Photometry Sonifier
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-mono">
                WEB AUDIO SYNTH
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">
              Target: <span className="text-[#8CE8B5] font-bold">{starName}</span> // Continuous Flux-to-Pitch Inversion
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Audio Download Button */}
          <button
            onClick={handleDownloadWav}
            title="Download Sonified WAV File"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-transparent hover:bg-white/[0.08] border border-white/15 text-slate-200 hover:text-white text-xs font-mono font-bold transition-all shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span className="hidden sm:inline">Export WAV</span>
          </button>

          {onClose && (
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-lg bg-transparent hover:bg-white/[0.08] border border-white/15 text-slate-400 hover:text-slate-200 flex items-center justify-center font-mono text-xs"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* ─── Waveform & Visual Playhead Canvas ─── */}
      <div className="relative w-full rounded-xl overflow-hidden border border-white/15 bg-black/60">
        <canvas
          ref={canvasRef}
          width={640}
          height={130}
          className="w-full h-32 block cursor-pointer"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const clickProg = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
            handleScrub(clickProg)
          }}
        />

        {/* Phase Scale Labels Overlay */}
        <div className="absolute bottom-1 left-2 right-2 flex justify-between text-[9px] font-mono text-slate-400 pointer-events-none">
          <span>-0.15φ (Ingress)</span>
          <span className="text-solar-amber">0.00φ (Mid-Transit)</span>
          <span>+0.15φ (Egress)</span>
        </div>

        {/* Live Audio Visualizer Spectrum Bars Overlay */}
        {isPlaying && (
          <div className="absolute top-2 right-3 flex items-end gap-1 h-5 pointer-events-none">
            {[40, 75, 100, 60, 85, 30].map((h, i) => (
              <span
                key={i}
                className="w-1 bg-[#8CE8B5] rounded-full animate-pulse"
                style={{
                  height: `${h}%`,
                  animationDuration: `${0.3 + (i % 3) * 0.2}s`
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* ─── Scrubber Bar ─── */}
      <div className="flex flex-col gap-1.5">
        <div className="flex justify-between items-center text-xs font-mono">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Radio className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span className={phaseStatus.color}>{phaseStatus.text}</span>
          </span>
          <span className="text-slate-300 font-bold">
            Phase: <span className="text-[#8CE8B5]">{((-0.15 + progress * 0.30)).toFixed(4)} φ</span>
          </span>
        </div>

        <input
          type="range"
          min="0"
          max="1"
          step="0.002"
          value={progress}
          onChange={(e) => handleScrub(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5] hover:accent-[#a3f0c7] transition-all"
        />
      </div>

      {/* ─── Instantaneous Acoustic Telemetry Grid ─── */}
      <div className="grid grid-cols-3 gap-2.5">
        <div className="p-3 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono block">AUDIO PITCH</span>
          <div className="flex items-center justify-center gap-1.5 mt-0.5">
            <span className="text-lg font-bold font-mono text-[#8CE8B5]">
              {currentHz.toFixed(1)}
            </span>
            <span className="text-xs text-slate-500 font-mono">Hz</span>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono block">ACOUSTIC DIP</span>
          <div className="flex items-center justify-center gap-1.5 mt-0.5">
            <span className={`text-lg font-bold font-mono ${pitchDrop > 5 ? 'text-solar-amber' : 'text-slate-400'}`}>
              -{pitchDrop.toFixed(1)}
            </span>
            <span className="text-xs text-slate-500 font-mono">Hz</span>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono block">SPATIAL STEREO PAN</span>
          <div className="text-sm font-bold font-mono text-slate-200 mt-1">
            {progress < 0.45 ? 'LEFT (Ingress)' : progress > 0.55 ? 'RIGHT (Egress)' : 'CENTER (Transit)'}
          </div>
        </div>
      </div>

      {/* ─── Playback Controls & Settings ─── */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-white/10">
        {/* Play / Loop / Reset Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleTogglePlay}
            disabled={loading}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-mono font-bold transition-all cursor-pointer ${
              isPlaying
                ? 'bg-solar-amber text-slate-950 shadow-[0_0_15px_rgba(255,138,0,0.5)]'
                : 'bg-[#8CE8B5] hover:bg-[#a3f0c7] text-slate-950 shadow-[0_0_15px_rgba(140,232,181,0.4)]'
            }`}
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4 fill-current" />
                <span>PAUSE</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>PLAY TRANSIT</span>
              </>
            )}
          </button>

          <button
            onClick={() => handleScrub(0)}
            className="p-2 rounded-xl bg-transparent hover:bg-white/[0.06] border border-white/15 text-slate-400 hover:text-slate-200"
            title="Rewind to Ingress"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={() => setIsLooping(!isLooping)}
            className={`px-3 py-2 rounded-xl text-xs font-mono border transition-all ${
              isLooping
                ? 'bg-white/15 border-[#8CE8B5]/50 text-[#8CE8B5] shadow-[0_0_10px_rgba(140,232,181,0.2)]'
                : 'bg-transparent border-white/15 text-slate-400'
            }`}
            title="Toggle Continuous Loop"
          >
            LOOP: {isLooping ? 'ON' : 'OFF'}
          </button>
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-black/40 border border-white/15">
          <span className="text-[10px] font-mono text-slate-400 pl-2">SPEED:</span>
          {[0.5, 1.0, 2.0].map((s) => (
            <button
              key={s}
              onClick={() => setPlaybackSpeed(s)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-all ${
                playbackSpeed === s ? 'bg-white/15 text-[#8CE8B5] font-bold border border-[#8CE8B5]/40' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Volume & Carrier Presets */}
        <div className="flex items-center gap-3">
          {/* Base Pitch Preset */}
          <div className="flex items-center gap-1 p-1 rounded-xl bg-black/40 border border-white/15">
            <span className="text-[10px] font-mono text-slate-400 pl-1.5">PITCH:</span>
            {[
              { label: '220Hz', val: 220 },
              { label: '440Hz', val: 440 },
              { label: '660Hz', val: 660 },
            ].map((p) => (
              <button
                key={p.val}
                onClick={() => setBaseFreq(p.val)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-all ${
                  baseFreq === p.val ? 'bg-white/15 text-[#8CE8B5] font-bold border border-[#8CE8B5]/40' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Volume Control */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleVolumeChange(isMuted ? 0.7 : 0)}
              className="text-slate-400 hover:text-slate-200"
            >
              {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4 text-[#8CE8B5]" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={isMuted ? 0 : volume}
              onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
              className="w-16 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
