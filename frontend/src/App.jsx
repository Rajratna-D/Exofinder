import React, { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Orbit,
  Activity,
  Cpu,
  Radio,
  Database,
  Search,
  ShieldCheck,
  Waves,
  ArrowLeft,
  SlidersHorizontal,
  X,
  ExternalLink
} from 'lucide-react'
import LandingPortal from './components/LandingPortal'
import CelestialChamber from './components/CelestialChamber'
import PhotometryLab from './components/PhotometryLab'
import TargetCatalog from './components/TargetCatalog'
import AIDiagnostics from './components/AIDiagnostics'
import SonificationPlayer from './components/SonificationPlayer'

function App() {
  // Navigation & View Mode State
  const [viewMode, setViewMode] = useState('portal') // 'portal' | 'dashboard'
  const [activeTab, setActiveTab] = useState('chamber') // 'chamber' | 'photometry' | 'targets' | 'diagnostics'

  // Backend Telemetry State
  const [health, setHealth] = useState(null)
  const [targets, setTargets] = useState([])
  const [selectedTarget, setSelectedTarget] = useState(null)
  const [targetDetails, setTargetDetails] = useState(null)

  // Modals & Slide-out Drawers
  const [showAudioModal, setShowAudioModal] = useState(false)
  const [showDossierDrawer, setShowDossierDrawer] = useState(false)

  // Fetch initial telemetry and targets from FastAPI backend
  useEffect(() => {
    async function loadData() {
      try {
        const [healthRes, targetsRes] = await Promise.all([
          fetch('/api/health').then(r => r.json()),
          fetch('/api/targets?page=1&page_size=8&sort_by=transit_snr&sort_order=desc').then(r => r.json())
        ])
        setHealth(healthRes)
        setTargets(targetsRes.targets || [])
        if (targetsRes.targets && targetsRes.targets.length > 0) {
          const first = targetsRes.targets[0]
          setSelectedTarget(first)
          const det = await fetch(`/api/target/${first.star_id}`).then(r => r.json())
          setTargetDetails(det)
        }
      } catch (err) {
        console.error("Failed to connect to ExoFinder backend:", err)
      }
    }
    loadData()
  }, [])

  // Switch selected target
  const handleSelectTarget = async (t) => {
    setSelectedTarget(t)
    try {
      const det = await fetch(`/api/target/${t.star_id}`).then(r => r.json())
      setTargetDetails(det)
    } catch (e) {
      console.error("Error fetching target details:", e)
    }
  }

  // Active target parameters for 3D simulator
  const activePeriod = selectedTarget?.bls_period || 3.35
  const activeDepth = selectedTarget?.bls_depth || 0.0135
  const activeDuration = selectedTarget?.bls_duration || 0.09
  const derivedRadiusRatio = Math.min(0.20, Math.max(0.03, Math.sqrt(Math.max(0.0001, activeDepth))))

  // Dynamic Animated View Router (Smooth Cinematic Transition between Portal and Dashboard)
  return (
    <>
      {/* ─── Layer 1: Space Landing Portal (Rendered ONCE — Stays Mounted) ─── */}
      <div
        className={`fixed inset-0 z-30 transition-all duration-500 ease-in-out ${
          viewMode === 'portal'
            ? 'opacity-100 pointer-events-auto visible'
            : 'opacity-0 pointer-events-none invisible scale-[1.02]'
        }`}
        style={{ willChange: 'opacity, transform' }}
      >
        <LandingPortal
          isPortalActive={viewMode === 'portal'}
          onEnterDashboard={(tab = 'chamber') => {
            setActiveTab(tab)
            setViewMode('dashboard')
          }}
          onOpenCatalog={() => {
            setActiveTab('targets')
            setViewMode('dashboard')
          }}
          onOpenSonifier={() => setShowAudioModal(true)}
          targetCount={health?.targets_indexed || 657}
        />
      </div>

      {/* ─── Layer 2: Science Exploration Dashboard & 3D Chamber (Rendered ONCE — Stays Mounted) ─── */}
      <div
        className={`relative min-h-screen bg-[#030408] text-slate-100 font-[family-name:var(--font-display)] flex flex-col pb-8 overflow-x-hidden select-none transition-all duration-500 ease-out ${
          viewMode === 'dashboard'
            ? 'opacity-100 pointer-events-auto scale-100'
            : 'opacity-0 pointer-events-none h-screen overflow-hidden scale-[0.98]'
        }`}
        style={{ willChange: 'opacity, transform' }}
      >
      {/* ─── De-Cluttered Minimalist Top Header Bar (Matching Landing Page Theme) ─── */}
      <header className="relative z-10 flex items-center justify-between px-6 py-3.5 mx-4 mt-4 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.4)]">
        {/* Left: Back to Portal & Brand */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setViewMode('portal')}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-transparent hover:bg-white/[0.08] text-slate-300 hover:text-white border border-white/15 hover:border-white/30 text-xs font-['JetBrains_Mono',monospace] transition-all"
            title="Return to Space Portal"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span className="hidden sm:inline">PORTAL</span>
          </button>

          <div className="h-5 w-px bg-white/15 mx-1 hidden sm:block" />

          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-[#8CE8B5] shadow-[0_0_12px_rgba(140,232,181,0.15)]">
              <Orbit className="w-4 h-4 animate-[spin_16s_linear_infinite]" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-['Syne',sans-serif] font-[900] text-sm tracking-wider text-white">
                EXOFINDER
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full border border-[#8CE8B5]/30 text-[#8CE8B5] font-['JetBrains_Mono',monospace] font-semibold bg-transparent">
                NOMINAL
              </span>
            </div>
          </div>
        </div>

        {/* Center: Spacious Workspace Mode Tabs */}
        <nav className="flex items-center gap-1.5 p-1 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md">
          <button
            onClick={() => setActiveTab('chamber')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-chamber font-bold tracking-wider uppercase transition-all ${
              activeTab === 'chamber'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_15px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Orbit className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>3D CHAMBER</span>
          </button>
          <button
            onClick={() => setActiveTab('photometry')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-['Space_Grotesk',sans-serif] font-bold tracking-wider uppercase transition-all ${
              activeTab === 'photometry'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_15px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Activity className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>PHOTOMETRY LAB</span>
          </button>
          <button
            onClick={() => setActiveTab('targets')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-['Space_Grotesk',sans-serif] font-bold tracking-wider uppercase transition-all ${
              activeTab === 'targets'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_15px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Database className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>CATALOG ({health?.targets_indexed || 657})</span>
          </button>
          <button
            onClick={() => setActiveTab('diagnostics')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-['Space_Grotesk',sans-serif] font-bold tracking-wider uppercase transition-all ${
              activeTab === 'diagnostics'
                ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_15px_rgba(140,232,181,0.2)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span>AI DIAGNOSTICS</span>
          </button>
        </nav>

        {/* Right: Active Target & Action Tools */}
        <div className="flex items-center gap-2.5">
          {/* Active Target Indicator */}
          <div className="hidden md:flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-transparent border border-white/15 text-xs font-['JetBrains_Mono',monospace] backdrop-blur-md">
            <span className="w-1.5 h-1.5 rounded-full bg-[#8CE8B5] animate-ping" />
            <span className="text-slate-400">TARGET:</span>
            <span className="text-[#8CE8B5] font-bold">{selectedTarget ? selectedTarget.tic_id : "TIC 77202722"}</span>
          </div>

          {/* Audio Sonifier Launcher */}
          <button
            onClick={() => setShowAudioModal(true)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-transparent hover:bg-white/[0.08] border border-white/15 text-slate-200 font-['JetBrains_Mono',monospace] text-xs font-bold transition-all shadow-sm"
            title="Acoustic Transit Sonifier"
          >
            <Waves className="w-3.5 h-3.5 text-[#8CE8B5] animate-pulse" />
            <span className="hidden lg:inline">SONIFY</span>
          </button>

          {/* Slide-out AI Dossier Toggle Button */}
          <button
            onClick={() => setShowDossierDrawer(!showDossierDrawer)}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border text-xs font-['JetBrains_Mono',monospace] font-bold transition-all ${
              showDossierDrawer
                ? 'bg-white/15 border-[#8CE8B5]/60 text-white shadow-[0_0_15px_rgba(140,232,181,0.25)]'
                : 'bg-transparent hover:bg-white/[0.08] border-white/15 text-slate-300'
            }`}
            title="Toggle AI Intelligence Dossier Drawer"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-[#8CE8B5]" />
            <span className="hidden sm:inline">AI DOSSIER</span>
          </button>
        </div>
      </header>

      {/* ─── Full-Width Main Stage with Smooth Persistent Tab Transitions ─── */}
      <main className="max-w-[1560px] mx-auto w-full px-4 mt-4 flex-1 relative min-h-[660px]">
        <div className="tab-stage-grid">
          {/* Tab 1: 3D Keplerian Celestial Chamber (Rendered ONCE — Smooth GPU Crossfade) */}
          <div
            className={`tab-stage-panel ${
              activeTab === 'chamber' ? 'tab-stage-active' : 'tab-stage-inactive'
            }`}
          >
            <div className="flex flex-col gap-4">
              {/* 3D WebGL Keplerian Chamber View */}
              <div className="p-6 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-4">
                <CelestialChamber
                  isActive={viewMode === 'dashboard' && activeTab === 'chamber'}
                  starName={selectedTarget?.tic_id || "TIC 77202722"}
                  defaultPeriod={activePeriod}
                  defaultDepth={activeDepth}
                  defaultDuration={activeDuration}
                  defaultRadiusRatio={derivedRadiusRatio}
                  defaultInclination={88.5}
                />

                {/* Minimalist Floating Telemetry Stat Strip (Completely Transparent Cards) */}
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
                  {[
                    { label: 'ORBITAL PERIOD', value: `${activePeriod.toFixed(3)} d`, color: 'text-white' },
                    { label: 'TRANSIT DEPTH', value: `${(activeDepth * 100).toFixed(3)}%`, color: 'text-[#8CE8B5]' },
                    { label: 'TRANSIT DURATION', value: `${(activeDuration * 24).toFixed(1)} hrs`, color: 'text-white' },
                    { label: 'TRANSIT SNR', value: `${selectedTarget?.transit_snr ? selectedTarget.transit_snr.toFixed(1) : "28.4"} σ`, color: 'text-[#8CE8B5]' },
                    { label: 'PLANET RADIUS', value: targetDetails?.planetary_physics?.planet_radius_earth ? `${targetDetails.planetary_physics.planet_radius_earth} R⊕` : `${(derivedRadiusRatio * 109.2).toFixed(2)} R⊕`, color: 'text-white' },
                    { label: 'SEMI-MAJOR AXIS', value: targetDetails?.planetary_physics?.semi_major_axis_au ? `${targetDetails.planetary_physics.semi_major_axis_au} AU` : `${(Math.pow(activePeriod / 365.25, 2/3)).toFixed(3)} AU`, color: 'text-[#8CE8B5]' },
                    { label: 'EQUILIBRIUM TEMP', value: targetDetails?.planetary_physics?.equilibrium_temperature_k ? `${targetDetails.planetary_physics.equilibrium_temperature_k} K` : `${Math.round(278.5 / Math.sqrt(Math.max(0.01, Math.pow(activePeriod / 365.25, 2/3))))} K`, color: 'text-white' },
                    { label: 'HABITABLE ZONE', value: targetDetails?.planetary_physics?.habitable_zone_label || (activePeriod > 100 ? 'Goldilocks' : 'Too Hot'), color: 'text-[#8CE8B5]' },
                  ].map((stat) => (
                    <div key={stat.label} className="p-3.5 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
                      <p className={`font-chamber text-xl font-bold ${stat.color} truncate`}>
                        {stat.value}
                      </p>
                      <p className="font-chamber text-[10px] font-bold text-slate-400 mt-1 tracking-[0.16em] uppercase truncate">
                        {stat.label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Star Selection Carousel Bar (Transparent Sci-Fi Style) */}
              <div className="p-5 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-[0.18em] font-chamber flex items-center gap-2">
                    <Search className="w-3.5 h-3.5 text-[#8CE8B5]" />
                    Featured Stellar Systems
                  </span>
                  <span className="text-[10px] text-slate-400 font-chamber">
                    SELECT TARGET TO MOUNT ORBITAL PHYSICS
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {targets.slice(0, 4).map((t) => {
                    const isSelected = selectedTarget?.star_id === t.star_id
                    return (
                      <button
                        key={t.star_id}
                        onClick={() => handleSelectTarget(t)}
                        className={`p-3.5 rounded-xl text-left transition-all border cursor-pointer ${
                          isSelected
                            ? 'bg-white/[0.06] border-[#8CE8B5]/50 shadow-[0_0_15px_rgba(140,232,181,0.2)]'
                            : 'bg-transparent hover:bg-white/[0.03] border-white/10 hover:border-white/20'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-chamber text-xs font-bold text-white">{t.tic_id}</span>
                          <span className={`text-[9px] px-2 py-0.5 rounded font-chamber font-bold ${
                            t.predicted_label === 'confirmed_planet'
                              ? 'bg-transparent text-[#8CE8B5] border border-[#8CE8B5]/40'
                              : t.predicted_label === 'eclipsing_binary'
                              ? 'bg-transparent text-amber-400 border border-amber-500/30'
                              : 'bg-transparent text-slate-400 border border-white/10'
                          }`}>
                            {t.predicted_label === 'confirmed_planet' ? 'PLANET' : t.predicted_label === 'eclipsing_binary' ? 'BINARY' : 'FP'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center mt-2.5 text-xs font-chamber text-slate-400">
                          <span>P: {t.bls_period ? `${t.bls_period.toFixed(2)}d` : '--'}</span>
                          <span className="text-[#8CE8B5] font-bold">{t.confidence ? `${(t.confidence * 100).toFixed(0)}%` : '--'}</span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Tab 2: Photometry Lab (Rendered ONCE — Smooth GPU Crossfade) */}
          <div
            className={`tab-stage-panel ${
              activeTab === 'photometry' ? 'tab-stage-active' : 'tab-stage-inactive'
            }`}
          >
            <PhotometryLab
              starId={selectedTarget?.star_id || 77202722}
              starName={selectedTarget?.tic_id || "TIC 77202722"}
            />
          </div>

          {/* Tab 3: Target Catalog (Rendered ONCE — Smooth GPU Crossfade) */}
          <div
            className={`tab-stage-panel ${
              activeTab === 'targets' ? 'tab-stage-active' : 'tab-stage-inactive'
            }`}
          >
            <TargetCatalog
              currentTargetId={selectedTarget?.star_id}
              onSelectTarget={(t) => {
                handleSelectTarget(t)
                setActiveTab('chamber')
              }}
            />
          </div>

          {/* Tab 4: AI Diagnostics (Rendered ONCE — Smooth GPU Crossfade) */}
          <div
            className={`tab-stage-panel ${
              activeTab === 'diagnostics' ? 'tab-stage-active' : 'tab-stage-inactive'
            }`}
          >
            <AIDiagnostics
              target={selectedTarget}
              targetDetails={targetDetails}
            />
          </div>
        </div>
      </main>

      {/* ─── Slide-Out AI Intelligence Dossier Drawer ─── */}
      {showDossierDrawer && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop Overlay */}
          <div
            onClick={() => setShowDossierDrawer(false)}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
          />

          {/* Drawer Sidebar */}
          <aside className="relative w-full max-w-md bg-void-light border-l border-cyan-500/30 p-6 flex flex-col gap-5 overflow-y-auto shadow-[-10px_0_30px_rgba(0,0,0,0.8)] z-10 animate-[slideIn_0.2s_ease-out]">
            {/* Drawer Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-purple-950/80 border border-purple-500/40 flex items-center justify-center text-purple-400">
                  <ShieldCheck className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="font-mono text-sm font-bold text-slate-100 uppercase tracking-wider">
                    AI Intelligence Dossier
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Target: <span className="text-cyan-300 font-bold">{selectedTarget?.tic_id}</span>
                  </span>
                </div>
              </div>
              <button
                onClick={() => setShowDossierDrawer(false)}
                className="w-7 h-7 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-200 flex items-center justify-center font-mono text-xs"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Model Consensus Classification */}
            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
              <div className="flex justify-between items-center">
                <span className="text-[11px] font-mono text-slate-400 uppercase">Stacking ML Prediction</span>
                <span className="text-[10px] font-mono text-cyan-400">RF + XGB + LR</span>
              </div>
              <div className="text-xl font-bold text-emerald-400 uppercase tracking-wide">
                {selectedTarget?.predicted_label ? selectedTarget.predicted_label.replace('_', ' ') : "Confirmed Planet"}
              </div>
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Confidence Score:</span>
                <span className="text-solar-amber font-bold">
                  {selectedTarget?.confidence ? `${(selectedTarget.confidence * 100).toFixed(1)}%` : "94.2%"}
                </span>
              </div>

              {/* Probability Breakdown Bars */}
              <div className="flex flex-col gap-2 mt-2 pt-2 border-t border-slate-800/80">
                {[
                  { label: 'Confirmed Planet', prob: targetDetails?.model_comparison?.stacking_ml?.confirmed_planet || 0.85, color: 'bg-emerald-500', text: 'text-emerald-400' },
                  { label: 'Eclipsing Binary', prob: targetDetails?.model_comparison?.stacking_ml?.eclipsing_binary || 0.08, color: 'bg-pink-500', text: 'text-pink-400' },
                  { label: 'False Positive', prob: targetDetails?.model_comparison?.stacking_ml?.false_positive || 0.07, color: 'bg-slate-500', text: 'text-slate-400' },
                ].map((p) => (
                  <div key={p.label} className="space-y-1">
                    <div className="flex justify-between text-[11px] font-mono">
                      <span className="text-slate-400">{p.label}</span>
                      <span className={`font-bold ${p.text}`}>{(p.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className={`h-full ${p.color} transition-all duration-500`} style={{ width: `${p.prob * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Astrophysical Vetting Diagnostic Tests */}
            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-2">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                Astronomical Vetting Rules
              </h4>

              <div className="flex flex-col gap-2">
                {[
                  {
                    label: 'Transit SNR (≥ 5.0σ)',
                    value: targetDetails?.vetting_diagnostics?.snr_test?.value || `${selectedTarget?.transit_snr?.toFixed(1) || "28.4"}σ`,
                    passed: targetDetails?.vetting_diagnostics?.snr_test?.passed ?? true
                  },
                  {
                    label: 'Odd/Even Depth (< 3.0σ)',
                    value: targetDetails?.vetting_diagnostics?.odd_even_test?.value || '0.42σ',
                    passed: targetDetails?.vetting_diagnostics?.odd_even_test?.passed ?? true
                  },
                  {
                    label: 'Secondary Eclipse Absence',
                    value: targetDetails?.vetting_diagnostics?.secondary_eclipse_test?.value || '0.0002',
                    passed: targetDetails?.vetting_diagnostics?.secondary_eclipse_test?.passed ?? true
                  },
                  {
                    label: 'U-Shape Ratio (≥ 0.65)',
                    value: targetDetails?.vetting_diagnostics?.u_shape_test?.value || '0.78',
                    passed: targetDetails?.vetting_diagnostics?.u_shape_test?.passed ?? true
                  }
                ].map((check) => (
                  <div
                    key={check.label}
                    className={`p-2.5 rounded-xl border flex items-center justify-between ${
                      check.passed
                        ? 'bg-emerald-950/20 border-emerald-500/30'
                        : 'bg-rose-950/20 border-rose-500/30'
                    }`}
                  >
                    <div className="flex flex-col">
                      <span className="text-[11px] font-mono text-slate-300">{check.label}</span>
                      <span className="text-[9px] font-mono text-slate-500">
                        {check.passed ? 'PASSED CRITERIA' : 'FAILED CRITERIA'}
                      </span>
                    </div>
                    <span className={`font-mono text-xs font-bold ${check.passed ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {check.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Keplerian Astrophysical Parameters (Roadmap Item A) */}
            <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-2">
                <Orbit className="w-3.5 h-3.5 text-cyan-400" />
                Planetary Astrophysics & Habitable Zone
              </h4>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">PLANET RADIUS</span>
                  <span className="text-emerald-400 font-bold">
                    {targetDetails?.planetary_physics?.planet_radius_earth ? `${targetDetails.planetary_physics.planet_radius_earth} R⊕` : `${(derivedRadiusRatio * 109.2).toFixed(1)} R⊕`}
                  </span>
                  <span className="text-[9px] text-slate-500 block">
                    {targetDetails?.planetary_physics?.planet_classification || "Terrestrial / Exoplanet"}
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">SEMI-MAJOR AXIS (a)</span>
                  <span className="text-cyan-400 font-bold">
                    {targetDetails?.planetary_physics?.semi_major_axis_au ? `${targetDetails.planetary_physics.semi_major_axis_au} AU` : `${(Math.pow(activePeriod / 365.25, 2/3)).toFixed(3)} AU`}
                  </span>
                  <span className="text-[9px] text-slate-500 block">
                    v: {targetDetails?.planetary_physics?.orbital_velocity_kms || (29.78 / Math.sqrt(Math.max(0.01, Math.pow(activePeriod / 365.25, 2/3)))).toFixed(1)} km/s
                  </span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">EQUILIBRIUM TEMP (T_eq)</span>
                  <span className="text-amber-400 font-bold">
                    {targetDetails?.planetary_physics?.equilibrium_temperature_k ? `${targetDetails.planetary_physics.equilibrium_temperature_k} K` : `${Math.round(278.5 / Math.sqrt(Math.max(0.01, Math.pow(activePeriod / 365.25, 2/3))))} K`}
                  </span>
                  <span className="text-[9px] text-slate-500 block">Bond Albedo 0.3</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">HABITABLE ZONE</span>
                  <span className="text-purple-400 font-bold text-[11px] truncate block">
                    {targetDetails?.planetary_physics?.habitable_zone_label || (activePeriod > 100 ? "Goldilocks Zone" : "Inner Scorched")}
                  </span>
                  <span className="text-[9px] text-slate-500 block">Kasting-Shields</span>
                </div>
              </div>
            </div>

            {/* Quick Diagnostic Jump CTA */}
            <button
              onClick={() => {
                setShowDossierDrawer(false)
                setActiveTab('diagnostics')
              }}
              className="mt-auto flex items-center justify-center gap-2 p-3 rounded-xl bg-purple-950/70 hover:bg-purple-900/80 border border-purple-500/40 text-purple-300 text-xs font-mono font-bold transition-all shadow-[0_0_15px_rgba(168,85,247,0.2)]"
            >
              <span>OPEN FULL AI DIAGNOSTICS LAB</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          </aside>
        </div>
      )}
      </div>

      {/* ─── Global Audio Sonification Modal ─── */}
      {showAudioModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-2xl animate-[fadeIn_0.15s_ease-out]">
            <SonificationPlayer
              starId={selectedTarget?.star_id || 77202722}
              starName={selectedTarget?.tic_id || "TIC 77202722"}
              onClose={() => setShowAudioModal(false)}
            />
          </div>
        </div>
      )}
    </>
  )
}

export default App
