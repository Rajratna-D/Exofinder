import React, { useState, useEffect, useMemo } from 'react'
import { Cpu, Sliders, ShieldCheck, RotateCcw, Activity, CheckCircle2, XCircle, Sparkles, Info } from 'lucide-react'

function AIDiagnostics({ target, targetDetails }) {
  // Base target parameters
  const starId = target?.star_id || 77202722
  const starName = target?.tic_id || "TIC 77202722"

  // ─── Interactive "What-If" Sensitivity Sliders State ───
  const [transitSnr, setTransitSnr] = useState(target?.transit_snr || 15.0)
  const [meanToMaxRatio, setMeanToMaxRatio] = useState(targetDetails?.features?.mean_to_max_depth_ratio?.value || 0.78)
  const [oddEvenSigma, setOddEvenSigma] = useState(targetDetails?.features?.odd_even_mismatch_sigma?.value || 0.42)
  const [secEclipse, setSecEclipse] = useState(targetDetails?.features?.secondary_eclipse_check?.value || 0.0002)
  const [blsDepth, setBlsDepth] = useState(target?.bls_depth || 0.0135)
  const [inTransitSkew, setInTransitSkew] = useState(targetDetails?.features?.in_transit_skew?.value || 0.12)

  // Real-time Inference Result from POST /api/classify
  const [livePrediction, setLivePrediction] = useState(null)
  const [isInferring, setIsInferring] = useState(false)

  // Sync sliders when active target changes
  useEffect(() => {
    if (target) {
      setTransitSnr(target.transit_snr || 15.0)
      setBlsDepth(target.bls_depth || 0.0135)
    }
    if (targetDetails?.features) {
      setMeanToMaxRatio(targetDetails.features.mean_to_max_depth_ratio?.value ?? 0.78)
      setOddEvenSigma(targetDetails.features.odd_even_mismatch_sigma?.value ?? 0.42)
      setSecEclipse(targetDetails.features.secondary_eclipse_check?.value ?? 0.0002)
      setInTransitSkew(targetDetails.features.in_transit_skew?.value ?? 0.12)
    }
  }, [target, targetDetails])

  // Reset to original target values
  const handleReset = () => {
    if (target) {
      setTransitSnr(target.transit_snr || 15.0)
      setBlsDepth(target.bls_depth || 0.0135)
    }
    if (targetDetails?.features) {
      setMeanToMaxRatio(targetDetails.features.mean_to_max_depth_ratio?.value ?? 0.78)
      setOddEvenSigma(targetDetails.features.odd_even_mismatch_sigma?.value ?? 0.42)
      setSecEclipse(targetDetails.features.secondary_eclipse_check?.value ?? 0.0002)
      setInTransitSkew(targetDetails.features.in_transit_skew?.value ?? 0.12)
    }
  }

  // Debounced real-time classification call to FastAPI backend
  useEffect(() => {
    const timer = setTimeout(async () => {
      setIsInferring(true)
      try {
        const payload = {
          star_id: starId,
          transit_snr: transitSnr,
          mean_to_max_depth_ratio: meanToMaxRatio,
          odd_even_mismatch_sigma: oddEvenSigma,
          secondary_eclipse_check: secEclipse,
          bls_depth: blsDepth,
          in_transit_skew: inTransitSkew
        }
        const res = await fetch('/api/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        if (res.ok) {
          const data = await res.json()
          setLivePrediction(data)
        }
      } catch (err) {
        console.error("Live inference failed:", err)
      } finally {
        setIsInferring(false)
      }
    }, 120) // 120ms debounce

    return () => clearTimeout(timer)
  }, [starId, transitSnr, meanToMaxRatio, oddEvenSigma, secEclipse, blsDepth, inTransitSkew])

  // Active probabilities
  const probs = livePrediction?.probabilities || {
    confirmed_planet: 0.85,
    eclipsing_binary: 0.08,
    false_positive: 0.07
  }

  const activeLabel = livePrediction?.predicted_label || target?.predicted_label || 'confirmed_planet'
  const activeConfidence = livePrediction?.confidence || target?.confidence || 0.85

  // Vetting checks calculation
  const vettingStatus = livePrediction?.vetting_status || {
    snr_test_passed: transitSnr >= 5.0,
    odd_even_passed: oddEvenSigma < 3.0,
    secondary_eclipse_passed: secEclipse < 0.005,
    u_shape_passed: meanToMaxRatio >= 0.65
  }

  const passCount = useMemo(() => {
    return [
      vettingStatus.snr_test_passed,
      vettingStatus.odd_even_passed,
      vettingStatus.secondary_eclipse_passed,
      vettingStatus.u_shape_passed
    ].filter(Boolean).length
  }, [vettingStatus])

  return (
    <div className="p-6 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-4 select-none">
      {/* ─── Header & Status ─── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-[#8CE8B5] shadow-[0_0_12px_rgba(140,232,181,0.15)]">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>AI Diagnostics &amp; Vetting Engine</span>
              <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-mono tracking-wider font-semibold">
                CALIBRATED STACKING ENSEMBLE
              </span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              ACTIVE PROBE: <span className="text-[#8CE8B5] font-bold">{starName}</span> &bull; ARCHITECTURE: <span className="text-white font-bold">XGBOOST + RANDOM FOREST</span>
            </p>
          </div>
        </div>

        {/* Live Inference Indicator & Reset Button */}
        <div className="flex items-center gap-2.5">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-mono font-bold transition-all ${
            isInferring
              ? 'bg-amber-400/10 border-amber-400/50 text-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.2)]'
              : 'bg-white/[0.04] border-[#8CE8B5]/40 text-[#8CE8B5]'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isInferring ? 'bg-amber-400 animate-ping' : 'bg-[#8CE8B5]'}`} />
            <span>{isInferring ? 'INFERRING...' : 'MODEL SYNCHRONIZED'}</span>
          </div>

          <button
            onClick={handleReset}
            title="Reset all perturbation sliders to original stellar parameters"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-transparent hover:bg-white/[0.08] border border-white/15 text-slate-200 hover:text-white text-xs font-mono font-bold transition-all cursor-pointer shadow-sm"
          >
            <RotateCcw className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>Reset Sliders</span>
          </button>
        </div>
      </div>

      {/* ─── Top Telemetry Summary (4 Metric Cards) ─── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Card 1: Astronomical Classification */}
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">ENSEMBLE VERDICT</span>
          <span className={`text-xl sm:text-2xl font-bold font-mono mt-1 block uppercase ${
            activeLabel === 'confirmed_planet'
              ? 'text-[#8CE8B5]'
              : activeLabel === 'eclipsing_binary'
              ? 'text-amber-400'
              : 'text-slate-300'
          }`}>
            {activeLabel === 'confirmed_planet' ? 'PLANET' : activeLabel === 'eclipsing_binary' ? 'BINARY' : 'FALSE POS'}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">PRIMARY AI CLASS</span>
        </div>

        {/* Card 2: Model Confidence */}
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">CONFIDENCE CERTAINTY</span>
          <span className="text-xl sm:text-2xl font-bold text-[#8CE8B5] font-mono mt-1 block">
            {(activeConfidence * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-[#8CE8B5]/70 font-mono mt-0.5 block">POSTERIOR PROBABILITY</span>
        </div>

        {/* Card 3: Vetting Pass Rate */}
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">VETTING PASS RATE</span>
          <span className={`text-xl sm:text-2xl font-bold font-mono mt-1 block ${
            passCount === 4 ? 'text-[#8CE8B5]' : passCount >= 2 ? 'text-amber-400' : 'text-rose-400'
          }`}>
            {passCount} / 4 PASSED
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">TESS VETTING CHECKS</span>
        </div>

        {/* Card 4: Primary Driver */}
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">PRIMARY DRIVER</span>
          <span className="text-xl sm:text-2xl font-bold text-white font-mono mt-1 block truncate px-1">
            {meanToMaxRatio >= 0.65 ? 'U-SHAPED' : 'V-SHAPED'}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">DOMINANT MORPHOLOGY</span>
        </div>
      </div>

      {/* ─── Main Decluttered Diagnostic Stage (2 Clean Spacious Panes) ─── */}
      <div className="grid grid-cols-12 gap-4">
        {/* Left Column (5 cols): Probability Breakdown & Automated Vetting Tests */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-4">
          {/* Card: Probability Distribution */}
          <div className="p-5 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md flex flex-col gap-3.5">
            <div className="flex items-center justify-between pb-2 border-b border-white/10">
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-[#8CE8B5]" />
                Class Probability Distribution
              </span>
              <span className="text-[10px] font-mono text-slate-400">SOFTMAX OUTPUT</span>
            </div>

            <div className="flex flex-col gap-3 pt-1">
              {[
                {
                  label: 'Confirmed Exoplanet',
                  prob: probs.confirmed_planet,
                  color: 'from-emerald-500 to-[#8CE8B5]',
                  textColor: 'text-[#8CE8B5]',
                  tag: 'TARGET CLASS'
                },
                {
                  label: 'Eclipsing Binary',
                  prob: probs.eclipsing_binary,
                  color: 'from-amber-500 to-amber-400',
                  textColor: 'text-amber-400',
                  tag: 'STELLAR'
                },
                {
                  label: 'False Positive / Noise',
                  prob: probs.false_positive,
                  color: 'from-slate-600 to-slate-400',
                  textColor: 'text-slate-300',
                  tag: 'INSTRUMENT'
                },
              ].map((p) => (
                <div key={p.label} className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300 font-medium">{p.label}</span>
                    <span className={`font-bold ${p.textColor}`}>{(p.prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${p.color} rounded-full transition-all duration-300`}
                      style={{ width: `${Math.max(1, p.prob * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Card: Automated Astrophysical Vetting Matrix */}
          <div className="p-5 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md flex flex-col gap-3">
            <div className="flex items-center justify-between pb-2 border-b border-white/10">
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5 text-[#8CE8B5]" />
                Automated Vetting Checklist
              </span>
              <span className="text-[10px] font-mono text-slate-400">TESS BENCHMARK</span>
            </div>

            <div className="grid grid-cols-1 gap-2 pt-1">
              {[
                {
                  name: 'Signal-to-Noise Ratio (SNR)',
                  criterion: '≥ 5.0σ',
                  value: `${transitSnr.toFixed(1)}σ`,
                  passed: vettingStatus.snr_test_passed
                },
                {
                  name: 'Odd / Even Depth Ratio',
                  criterion: '< 3.0σ',
                  value: `${oddEvenSigma.toFixed(2)}σ`,
                  passed: vettingStatus.odd_even_passed
                },
                {
                  name: 'Secondary Eclipse Occultation',
                  criterion: '< 0.0050',
                  value: `${secEclipse.toFixed(4)}`,
                  passed: vettingStatus.secondary_eclipse_passed
                },
                {
                  name: 'U-Shape vs V-Shape Profile',
                  criterion: '≥ 0.65',
                  value: `${meanToMaxRatio.toFixed(3)}`,
                  passed: vettingStatus.u_shape_passed
                }
              ].map((v) => (
                <div
                  key={v.name}
                  className={`p-2.5 rounded-xl border text-xs font-mono flex items-center justify-between transition-all ${
                    v.passed
                      ? 'bg-transparent border-white/10 hover:border-white/20 text-slate-200'
                      : 'bg-rose-950/20 border-rose-500/30 text-rose-300'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {v.passed ? (
                      <CheckCircle2 className="w-4 h-4 text-[#8CE8B5] shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    )}
                    <span className="font-medium text-slate-200">{v.name}</span>
                  </div>
                  <div className="flex items-center gap-2 text-right">
                    <span className="text-slate-400 text-[11px]">({v.criterion})</span>
                    <span className={`font-bold ${v.passed ? 'text-[#8CE8B5]' : 'text-rose-400'}`}>
                      {v.value}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column (7 cols): Clean, Spacious Sensitivity Simulator */}
        <div className="col-span-12 lg:col-span-7 flex flex-col gap-4">
          <div className="p-5 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md flex flex-col gap-4 h-full">
            <div className="flex items-center justify-between pb-2 border-b border-white/10">
              <span className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                <Sliders className="w-3.5 h-3.5 text-[#8CE8B5]" />
                Feature Perturbation Simulator ("What-If" Analysis)
              </span>
              <span className="text-[10px] font-mono text-slate-400">REAL-TIME STRESS TEST</span>
            </div>

            {/* Sliders Container (Clean, uncluttered, spacious) */}
            <div className="flex flex-col gap-4 py-1">
              {/* Slider 1: Transit SNR */}
              <div className="p-3 rounded-xl bg-transparent border border-white/10 hover:border-white/20 transition-all flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300 font-medium">1. Transit Signal-to-Noise Ratio (SNR)</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${
                    transitSnr >= 5.0 ? 'text-[#8CE8B5] bg-[#8CE8B5]/10' : 'text-rose-400 bg-rose-500/10'
                  }`}>
                    {transitSnr.toFixed(1)}σ {transitSnr >= 5.0 ? '✓ High SNR' : '⚠ Marginal'}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="35"
                  step="0.5"
                  value={transitSnr}
                  onChange={(e) => setTransitSnr(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.5σ (Noise fluke)</span>
                  <span className="text-[#8CE8B5]">5.0σ (NASA Threshold)</span>
                  <span>35.0σ (Deep Detection)</span>
                </div>
              </div>

              {/* Slider 2: U-Shape vs V-Shape Depth Ratio */}
              <div className="p-3 rounded-xl bg-transparent border border-white/10 hover:border-white/20 transition-all flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300 font-medium">2. Profile Shape (Mean-to-Max Depth Ratio)</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${
                    meanToMaxRatio >= 0.65 ? 'text-[#8CE8B5] bg-[#8CE8B5]/10' : 'text-amber-400 bg-amber-400/10'
                  }`}>
                    {meanToMaxRatio.toFixed(3)} {meanToMaxRatio >= 0.65 ? '✓ Flat U-Shape' : '⚠ Grazing V-Shape'}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.30"
                  max="0.95"
                  step="0.01"
                  value={meanToMaxRatio}
                  onChange={(e) => setMeanToMaxRatio(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.30 (V-Shape Binary)</span>
                  <span className="text-[#8CE8B5]">0.65 (Planet Threshold)</span>
                  <span>0.95 (Deep U-Shape)</span>
                </div>
              </div>

              {/* Slider 3: Odd/Even Transit Depth Mismatch */}
              <div className="p-3 rounded-xl bg-transparent border border-white/10 hover:border-white/20 transition-all flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300 font-medium">3. Odd / Even Depth Mismatch Significance</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${
                    oddEvenSigma < 3.0 ? 'text-[#8CE8B5] bg-[#8CE8B5]/10' : 'text-amber-400 bg-amber-400/10'
                  }`}>
                    {oddEvenSigma.toFixed(2)}σ {oddEvenSigma < 3.0 ? '✓ Consistent Depths' : '⚠ Alternating Binary'}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="8"
                  step="0.1"
                  value={oddEvenSigma}
                  onChange={(e) => setOddEvenSigma(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.0σ (Zero Mismatch)</span>
                  <span className="text-amber-400">3.0σ (Binary Threshold)</span>
                  <span>8.0σ (Clear Alternation)</span>
                </div>
              </div>

              {/* Slider 4: Secondary Eclipse Check */}
              <div className="p-3 rounded-xl bg-transparent border border-white/10 hover:border-white/20 transition-all flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300 font-medium">4. Secondary Eclipse Occultation Check</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${
                    secEclipse < 0.005 ? 'text-[#8CE8B5] bg-[#8CE8B5]/10' : 'text-rose-400 bg-rose-500/10'
                  }`}>
                    {secEclipse.toFixed(5)} {secEclipse < 0.005 ? '✓ No Secondary Dip' : '⚠ Luminous Secondary'}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0000"
                  max="0.0200"
                  step="0.0005"
                  value={secEclipse}
                  onChange={(e) => setSecEclipse(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.0000 (No Dip)</span>
                  <span className="text-rose-400">0.0050 (Stellar Eclipse)</span>
                  <span>0.0200 (Deep Secondary)</span>
                </div>
              </div>

              {/* Slider 5: Transit Depth */}
              <div className="p-3 rounded-xl bg-transparent border border-white/10 hover:border-white/20 transition-all flex flex-col gap-1.5">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-300 font-medium">5. Transit Depth (ΔF/F)</span>
                  <span className="px-2 py-0.5 rounded font-bold text-[#8CE8B5] bg-[#8CE8B5]/10">
                    {(blsDepth * 100).toFixed(3)}% ({Math.round(blsDepth * 1e6)} ppm)
                  </span>
                </div>
                <input
                  type="range"
                  min="0.001"
                  max="0.050"
                  step="0.001"
                  value={blsDepth}
                  onChange={(e) => setBlsDepth(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.1% (Super-Earth)</span>
                  <span className="text-[#8CE8B5]">1.0% (Jupiter-like)</span>
                  <span>5.0% (M-Dwarf Binary)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default React.memo(AIDiagnostics)
