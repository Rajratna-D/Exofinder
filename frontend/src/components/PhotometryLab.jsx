import React, { useState, useEffect, useRef } from 'react'
import { Activity, Layers, Download, RotateCcw, Cpu, Waves } from 'lucide-react'
import SonificationPlayer from './SonificationPlayer'

// Helper to draw clean axes and grid lines
function drawGrid(ctx, padding, plotW, plotH, minX, maxX, minY, maxY, xLabel, yLabel, type) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.10)'
  ctx.lineWidth = 1

  // Box border
  ctx.strokeRect(padding.left, padding.top, plotW, plotH)

  // X Axis ticks & grid
  const numXTicks = 6
  ctx.fillStyle = '#94a3b8'
  ctx.font = '10px "JetBrains Mono", monospace'
  ctx.textAlign = 'center'
  for (let i = 0; i <= numXTicks; i++) {
    const val = minX + (i / numXTicks) * (maxX - minX)
    const x = padding.left + (i / numXTicks) * plotW
    ctx.beginPath()
    ctx.moveTo(x, padding.top)
    ctx.lineTo(x, padding.top + plotH)
    ctx.stroke()
    ctx.fillText(val.toFixed(type === 'phase' ? 3 : 1), x, padding.top + plotH + 16)
  }

  // Y Axis ticks & grid
  const numYTicks = 5
  ctx.textAlign = 'right'
  for (let i = 0; i <= numYTicks; i++) {
    const val = minY + (i / numYTicks) * (maxY - minY)
    const y = padding.top + plotH - (i / numYTicks) * plotH
    ctx.beginPath()
    ctx.moveTo(padding.left, y)
    ctx.lineTo(padding.left + plotW, y)
    ctx.stroke()
    ctx.fillText(val.toFixed(3), padding.left - 8, y + 3)
  }

  // Axis Labels
  ctx.fillStyle = '#64748b'
  ctx.font = '11px "Space Grotesk", sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText(xLabel, padding.left + plotW / 2, padding.top + plotH + 30)

  ctx.save()
  ctx.translate(15, padding.top + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillText(yLabel, 0, 0)
  ctx.restore()
}

// Helper for CNN dual subplots
function drawSubplot(ctx, x, y, w, h, xData, yData, color, title) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'
  ctx.strokeRect(x, y, w, h)

  ctx.fillStyle = '#94a3b8'
  ctx.font = '10px "JetBrains Mono", monospace'
  ctx.textAlign = 'left'
  ctx.fillText(title, x + 8, y + 14)

  if (xData.length === 0 || yData.length === 0) return

  const minX = Math.min(...xData)
  const maxX = Math.max(...xData)
  const minY = Math.min(-0.015, Math.min(...yData) * 1.1)
  const maxY = Math.max(0.005, Math.max(...yData) * 1.1)

  const toSubX = (val) => x + ((val - minX) / (maxX - minX)) * w
  const toSubY = (val) => y + h - ((val - minY) / (maxY - minY)) * h

  // Zero-line (f - 1.0 = 0.0)
  const zeroY = toSubY(0.0)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'
  ctx.setLineDash([3, 3])
  ctx.beginPath()
  ctx.moveTo(x, zeroY)
  ctx.lineTo(x + w, zeroY)
  ctx.stroke()
  ctx.setLineDash([])

  // Data trace
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < xData.length; i++) {
    const px = toSubX(xData[i])
    const py = toSubY(yData[i])
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.stroke()

  // Binned points
  ctx.fillStyle = color
  for (let i = 0; i < xData.length; i++) {
    const px = toSubX(xData[i])
    const py = toSubY(yData[i])
    ctx.beginPath()
    ctx.arc(px, py, 2, 0, Math.PI * 2)
    ctx.fill()
  }
}

// Module-level lightcurve cache for instant 0ms switching between targets
const lightCurveCache = new Map()

// ─── High-Performance Canvas Chart for Time-Series & Phase-Folded Data ───
function PhotometryCanvas({
  data,
  type = 'timeseries', // 'timeseries', 'phase', 'cnn'
  showRaw = true,
  showDetrended = true,
  showBinned = true,
  showModel = true,
  zoomWindow = 0.2
}) {
  const containerRef = useRef(null)
  const canvasRef = useRef(null)
  const [hoverInfo, setHoverInfo] = useState(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // High DPI Support
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return

    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const w = rect.width
    const h = rect.height
    const padding = { top: 20, right: 25, bottom: 35, left: 55 }
    const plotW = w - padding.left - padding.right
    const plotH = h - padding.top - padding.bottom

    // Clear background
    ctx.clearRect(0, 0, w, h)

    if (!data) {
      ctx.fillStyle = '#64748b'
      ctx.font = '12px "JetBrains Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('NO PHOTOMETRIC DATA LOADED', w / 2, h / 2)
      return
    }

    // ─── Drawing Coordinate Systems ───
    if (type === 'timeseries') {
      const times = data.time_series?.time || []
      const detFlux = data.time_series?.detrended_flux || []
      const rawFlux = data.time_series?.raw_flux || []

      if (times.length === 0) return

      const minX = Math.min(...times)
      const maxX = Math.max(...times)
      
      // Calculate Y range with outlier clipping (99.5% percentile)
      let minY = 0.985
      let maxY = 1.015
      if (detFlux.length > 0) {
        const sorted = [...detFlux].filter(v => !isNaN(v)).sort((a, b) => a - b)
        const q01 = sorted[Math.floor(sorted.length * 0.005)] || 0.985
        const q99 = sorted[Math.floor(sorted.length * 0.995)] || 1.015
        minY = Math.min(0.98, q01 - 0.005)
        maxY = Math.max(1.02, q99 + 0.005)
      }

      const toX = (val) => padding.left + ((val - minX) / (maxX - minX)) * plotW
      const toY = (val) => padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH

      // Grid Lines (preserving JetBrains Mono & Space Grotesk fonts)
      drawGrid(ctx, padding, plotW, plotH, minX, maxX, minY, maxY, "Time (BTJD days)", "Norm. Flux", type)

      // 1. Raw Flux Scatter (Gray / Dim blue)
      if (showRaw && rawFlux.length > 0) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.28)'
        for (let i = 0; i < times.length; i++) {
          const px = toX(times[i])
          const py = toY(rawFlux[i])
          ctx.fillRect(px - 1, py - 1, 2, 2)
        }
      }

      // 2. Detrended Flux Scatter (Mint Green #8CE8B5)
      if (showDetrended && detFlux.length > 0) {
        ctx.fillStyle = '#8CE8B5'
        for (let i = 0; i < times.length; i++) {
          const px = toX(times[i])
          const py = toY(detFlux[i])
          ctx.fillRect(px - 1, py - 1, 2, 2)
        }
      }

      // 3. Normalized Baseline Reference Line
      const y1 = toY(1.0)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(padding.left, y1)
      ctx.lineTo(padding.left + plotW, y1)
      ctx.stroke()
      ctx.setLineDash([])
    } 
    else if (type === 'phase') {
      const phases = data.phase_folded?.phase || []
      const fluxes = data.phase_folded?.flux || []
      const binCenters = data.phase_folded?.bin_centers || []
      const binnedFlux = data.phase_folded?.binned_flux || []
      const theoCurve = data.phase_folded?.theoretical_curve || []

      const minX = -zoomWindow
      const maxX = zoomWindow
      
      // Calculate Y range focused on the transit dip
      let minY = 0.985
      let maxY = 1.01
      if (fluxes.length > 0) {
        const sorted = [...fluxes].filter(v => !isNaN(v)).sort((a, b) => a - b)
        const q01 = sorted[Math.floor(sorted.length * 0.005)] || 0.985
        const q99 = sorted[Math.floor(sorted.length * 0.995)] || 1.01
        minY = Math.min(0.975, q01 - 0.004)
        maxY = Math.max(1.015, q99 + 0.004)
      }

      const toX = (val) => padding.left + ((val - minX) / (maxX - minX)) * plotW
      const toY = (val) => padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH

      drawGrid(ctx, padding, plotW, plotH, minX, maxX, minY, maxY, "Orbital Phase (Φ)", "Norm. Flux", type)

      // 1. Individual Cadences (Mint Green Scatter)
      ctx.fillStyle = 'rgba(140, 232, 181, 0.40)'
      for (let i = 0; i < phases.length; i++) {
        const p = phases[i]
        if (p >= minX && p <= maxX) {
          const px = toX(p)
          const py = toY(fluxes[i])
          ctx.beginPath()
          ctx.arc(px, py, 1.8, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // 2. Theoretical Mandel & Agol Fit (Amber Dashed Line)
      if (showModel && theoCurve.length > 0 && binCenters.length === theoCurve.length) {
        ctx.strokeStyle = '#FFB300'
        ctx.lineWidth = 2.5
        ctx.setLineDash([5, 3])
        ctx.beginPath()
        let started = false
        for (let i = 0; i < binCenters.length; i++) {
          const bc = binCenters[i]
          if (bc >= minX && bc <= maxX) {
            const px = toX(bc)
            const py = toY(theoCurve[i])
            if (!started) {
              ctx.moveTo(px, py)
              started = true
            } else {
              ctx.lineTo(px, py)
            }
          }
        }
        ctx.stroke()
        ctx.setLineDash([])
      }

      // 3. Binned Median Profile (Rose Solid Line with Points)
      if (showBinned && binnedFlux.length > 0) {
        ctx.strokeStyle = '#F43F5E'
        ctx.lineWidth = 2.5
        ctx.beginPath()
        let started = false
        for (let i = 0; i < binCenters.length; i++) {
          const bc = binCenters[i]
          if (bc >= minX && bc <= maxX) {
            const px = toX(bc)
            const py = toY(binnedFlux[i])
            if (!started) {
              ctx.moveTo(px, py)
              started = true
            } else {
              ctx.lineTo(px, py)
            }
          }
        }
        ctx.stroke()

        // Binned point circles
        ctx.fillStyle = '#F43F5E'
        for (let i = 0; i < binCenters.length; i++) {
          const bc = binCenters[i]
          if (bc >= minX && bc <= maxX) {
            const px = toX(bc)
            const py = toY(binnedFlux[i])
            ctx.beginPath()
            ctx.arc(px, py, 2.5, 0, Math.PI * 2)
            ctx.fill()
          }
        }
      }

      // Ingress / Egress and Center Markers
      const x0 = toX(0.0)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(x0, padding.top)
      ctx.lineTo(x0, padding.top + plotH)
      ctx.stroke()
      ctx.setLineDash([])
    }
    else if (type === 'cnn') {
      const gView = data.cnn_views?.global_view || []
      const lView = data.cnn_views?.local_view || []
      const gPhase = data.cnn_views?.global_phase || []
      const lPhase = data.cnn_views?.local_phase || []

      // Draw two-panel split view
      const halfH = (plotH - 20) / 2

      // Panel 1: Global View (-0.5 to 0.5)
      drawSubplot(ctx, padding.left, padding.top, plotW, halfH, gPhase, gView, '#38BDF8', "NASA Astronet Global Conv1D (201 Bins)")

      // Panel 2: Local Zoomed View (-0.08 to 0.08)
      drawSubplot(ctx, padding.left, padding.top + halfH + 20, plotW, halfH, lPhase, lView, '#F43F5E', "NASA Astronet Local Zoomed Conv1D (61 Bins)")
    }

  }, [data, type, showRaw, showDetrended, showBinned, showModel, zoomWindow])

  // Mouse move handler for interactive telemetry crosshair
  const handleMouseMove = (e) => {
    if (!canvasRef.current || !data) return
    const rect = canvasRef.current.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top
    const padding = { top: 20, right: 25, bottom: 35, left: 55 }
    const plotW = rect.width - padding.left - padding.right
    const plotH = rect.height - padding.top - padding.bottom

    if (mouseX < padding.left || mouseX > padding.left + plotW || mouseY < padding.top || mouseY > padding.top + plotH) {
      setHoverInfo(null)
      return
    }

    if (type === 'timeseries') {
      const times = data.time_series?.time || []
      if (times.length === 0) return
      const minX = Math.min(...times)
      const maxX = Math.max(...times)
      const curX = minX + ((mouseX - padding.left) / plotW) * (maxX - minX)
      setHoverInfo({
        x: mouseX,
        y: mouseY,
        label: `T: ${curX.toFixed(3)} BTJD`
      })
    } else if (type === 'phase') {
      const minX = -zoomWindow
      const maxX = zoomWindow
      const curPhase = minX + ((mouseX - padding.left) / plotW) * (maxX - minX)
      setHoverInfo({
        x: mouseX,
        y: mouseY,
        label: `Phase Φ: ${curPhase >= 0 ? '+' : ''}${curPhase.toFixed(4)}`
      })
    } else {
      setHoverInfo(null)
    }
  }

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverInfo(null)}
      className="relative w-full h-[460px] bg-black/60 rounded-2xl overflow-hidden border border-white/15 backdrop-blur-md shadow-[0_12px_40px_rgba(0,0,0,0.6)] group cursor-crosshair"
    >
      <canvas ref={canvasRef} className="w-full h-full block" />

      {/* Interactive Telemetry Crosshair Hairline & Floating HUD Badge */}
      {hoverInfo && (
        <>
          <div
            className="absolute top-[20px] bottom-[35px] w-[1px] bg-[#8CE8B5]/50 pointer-events-none shadow-[0_0_8px_rgba(140,232,181,0.6)]"
            style={{ left: `${hoverInfo.x}px` }}
          />
          <div
            className="absolute top-3 px-3 py-1 rounded-lg bg-black/90 border border-[#8CE8B5]/40 text-[#8CE8B5] text-[11px] font-mono font-bold pointer-events-none shadow-[0_0_12px_rgba(0,0,0,0.8)] backdrop-blur-md"
            style={{
              left: `${Math.min(hoverInfo.x + 12, (containerRef.current?.clientWidth || 700) - 180)}px`,
            }}
          >
            {hoverInfo.label}
          </div>
        </>
      )}
    </div>
  )
}

// ─── Main Photometry Lab Component ───
function PhotometryLab({ starId = 77202722, starName = "TIC 77202722" }) {
  const [activeView, setActiveView] = useState('timeseries') // 'timeseries', 'phase', 'cnn'
  const [lcData, setLcData] = useState(lightCurveCache.get(starId) || null)
  const [loading, setLoading] = useState(!lightCurveCache.has(starId))
  const [error, setError] = useState(null)

  // Chart Trace Toggles
  const [showRaw, setShowRaw] = useState(true)
  const [showDetrended, setShowDetrended] = useState(true)
  const [showBinned, setShowBinned] = useState(true)
  const [showModel, setShowModel] = useState(true)
  const [zoomWindow, setZoomWindow] = useState(0.15) // Phase zoom window
  const [showSonifier, setShowSonifier] = useState(false)

  // Fetch lightcurve data on starId change with cache check
  useEffect(() => {
    let isMounted = true

    if (lightCurveCache.has(starId)) {
      setLcData(lightCurveCache.get(starId))
      setLoading(false)
      setError(null)
      return
    }

    async function fetchLC() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch(`/api/lightcurve/${starId}?max_points=2500`)
        if (!res.ok) {
          throw new Error(`Failed to load lightcurve for TIC ${starId}`)
        }
        const data = await res.json()
        lightCurveCache.set(starId, data)
        if (isMounted) {
          setLcData(data)
        }
      } catch (err) {
        console.error(err)
        if (isMounted) setError(err.message)
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchLC()

    return () => {
      isMounted = false
    }
  }, [starId])

  // Download Lightcurve Data as CSV
  const handleDownloadCSV = () => {
    if (!lcData?.time_series) return
    const times = lcData.time_series.time || []
    const detFlux = lcData.time_series.detrended_flux || []
    const rawFlux = lcData.time_series.raw_flux || []

    let csvContent = "data:text/csv;charset=utf-8,time,detrended_flux,raw_flux\n"
    for (let i = 0; i < times.length; i++) {
      csvContent += `${times[i]},${detFlux[i]},${rawFlux[i] || ""}\n`
    }
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", `lightcurve_TIC_${starId}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="p-6 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-4 select-none">
      {/* ─── Header & View Switcher ─── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-[#8CE8B5] shadow-[0_0_12px_rgba(140,232,181,0.15)]">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>Photometry Lab</span>
              <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-mono tracking-wider font-semibold">
                DUAL-STREAM ENGINE
              </span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              TARGET: <span className="text-[#8CE8B5] font-bold">{starName}</span> &bull; RECOVERED CADENCES: <span className="text-white font-bold">{lcData?.time_series?.time?.length || 0}</span>
            </p>
          </div>
        </div>

        {/* View Mode Pills */}
        <div className="flex items-center gap-1.5 p-1 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md self-start sm:self-auto">
          <button
            onClick={() => setActiveView('timeseries')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold uppercase tracking-wider transition-all flex items-center gap-1.5 ${
              activeView === 'timeseries'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>Time Series</span>
          </button>
          <button
            onClick={() => setActiveView('phase')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold uppercase tracking-wider transition-all flex items-center gap-1.5 ${
              activeView === 'phase'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <RotateCcw className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>Phase Folded</span>
          </button>
          <button
            onClick={() => setActiveView('cnn')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold uppercase tracking-wider transition-all flex items-center gap-1.5 ${
              activeView === 'cnn'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>1D-CNN Views</span>
          </button>
        </div>
      </div>

      {/* ─── Controls & Legend Bar ─── */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md text-xs font-mono">
        {/* Trace Toggles depending on active view */}
        <div className="flex items-center gap-4">
          {activeView === 'timeseries' && (
            <>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white transition-colors">
                <input
                  type="checkbox"
                  checked={showDetrended}
                  onChange={(e) => setShowDetrended(e.target.checked)}
                  className="accent-[#8CE8B5] rounded cursor-pointer"
                />
                <span className="text-[#8CE8B5] font-bold">Detrended Flux</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white transition-colors">
                <input
                  type="checkbox"
                  checked={showRaw}
                  onChange={(e) => setShowRaw(e.target.checked)}
                  className="accent-slate-400 rounded cursor-pointer"
                />
                <span className="text-slate-400">Raw Normalized</span>
              </label>
            </>
          )}

          {activeView === 'phase' && (
            <>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white transition-colors">
                <input
                  type="checkbox"
                  checked={showBinned}
                  onChange={(e) => setShowBinned(e.target.checked)}
                  className="accent-rose-500 rounded cursor-pointer"
                />
                <span className="text-rose-400 font-bold">Binned Median Profile</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300 hover:text-white transition-colors">
                <input
                  type="checkbox"
                  checked={showModel}
                  onChange={(e) => setShowModel(e.target.checked)}
                  className="accent-amber-400 rounded cursor-pointer"
                />
                <span className="text-amber-400 font-bold">Theoretical Transit Model</span>
              </label>

              {/* Phase Zoom Slider */}
              <div className="flex items-center gap-1.5 ml-4 pl-4 border-l border-white/10">
                <span className="text-slate-400 text-[11px]">Phase Window:</span>
                <button
                  onClick={() => setZoomWindow(0.08)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${zoomWindow === 0.08 ? 'bg-white/15 text-[#8CE8B5] border border-[#8CE8B5]/40 font-bold' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Zoom (±0.08)
                </button>
                <button
                  onClick={() => setZoomWindow(0.15)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${zoomWindow === 0.15 ? 'bg-white/15 text-[#8CE8B5] border border-[#8CE8B5]/40 font-bold' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Standard (±0.15)
                </button>
                <button
                  onClick={() => setZoomWindow(0.35)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${zoomWindow === 0.35 ? 'bg-white/15 text-[#8CE8B5] border border-[#8CE8B5]/40 font-bold' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  Wide (±0.35)
                </button>
              </div>
            </>
          )}

          {activeView === 'cnn' && (
            <span className="text-slate-400 flex items-center gap-2 text-xs">
              <span className="w-2 h-2 rounded-full bg-sky-400"></span> Global View (201 Bins)
              <span className="w-2 h-2 rounded-full bg-rose-500 ml-3"></span> Local View (61 Bins)
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Audio Sonifier Drawer Toggle */}
          <button
            onClick={() => setShowSonifier(!showSonifier)}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border text-xs font-mono font-bold transition-all ${
              showSonifier
                ? 'bg-white/15 border-[#8CE8B5]/60 text-white shadow-[0_0_12px_rgba(140,232,181,0.25)]'
                : 'bg-transparent hover:bg-white/[0.08] border border-white/15 text-[#8CE8B5]'
            }`}
          >
            <Waves className="w-3.5 h-3.5 text-[#8CE8B5] animate-pulse" />
            <span>{showSonifier ? 'Hide Sonifier' : 'Audio Sonifier'}</span>
          </button>

          {/* Export / Download CSV */}
          <button
            onClick={handleDownloadCSV}
            disabled={!lcData}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-transparent hover:bg-white/[0.08] border border-white/15 text-slate-200 hover:text-white text-xs font-mono font-bold transition-all disabled:opacity-40"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* ─── Expandable Sonification Player ─── */}
      {showSonifier && (
        <SonificationPlayer
          starId={starId}
          starName={starName}
          embedded={true}
          onClose={() => setShowSonifier(false)}
        />
      )}

      {/* ─── Photometric Canvas ─── */}
      {loading ? (
        <div className="w-full h-[460px] rounded-2xl bg-black/60 border border-white/15 backdrop-blur-md flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-[#8CE8B5] border-t-transparent animate-spin" />
          <span className="font-mono text-xs text-slate-400">DOWNLOADING STELLAR PHOTOMETRY FROM BACKEND...</span>
        </div>
      ) : error ? (
        <div className="w-full h-[460px] rounded-2xl bg-black/60 border border-rose-900/50 flex flex-col items-center justify-center gap-3">
          <span className="font-mono text-sm text-rose-400 font-bold">ERROR LOADING LIGHTCURVE</span>
          <span className="font-mono text-xs text-slate-400">{error}</span>
        </div>
      ) : (
        <PhotometryCanvas
          data={lcData}
          type={activeView}
          showRaw={showRaw}
          showDetrended={showDetrended}
          showBinned={showBinned}
          showModel={showModel}
          zoomWindow={zoomWindow}
        />
      )}

      {/* ─── Bottom Telemetry Strip ─── */}
      {lcData && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
          <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">RECOVERED PERIOD</span>
            <span className="text-lg sm:text-xl font-bold text-white font-mono mt-1 block">
              {lcData.period ? `${lcData.period.toFixed(5)} d` : '--'}
            </span>
          </div>
          <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">TRANSIT DEPTH (ΔF/F)</span>
            <span className="text-lg sm:text-xl font-bold text-[#8CE8B5] font-mono mt-1 block">
              {lcData.depth ? `${(lcData.depth * 100).toFixed(4)}%` : '--'}
            </span>
          </div>
          <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">TRANSIT DURATION</span>
            <span className="text-lg sm:text-xl font-bold text-white font-mono mt-1 block">
              {lcData.duration ? `${(lcData.duration * 24).toFixed(2)} hrs` : '--'}
            </span>
          </div>
          <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
            <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">EPOCH (T0)</span>
            <span className="text-lg sm:text-xl font-bold text-[#8CE8B5] font-mono mt-1 block">
              {lcData.epoch ? `${lcData.epoch.toFixed(4)} BTJD` : '0.0000'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default React.memo(PhotometryLab)
