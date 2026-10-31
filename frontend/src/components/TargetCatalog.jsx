import React, { useState, useEffect, useRef, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { Search, ArrowUpDown, ChevronRight, X, Database, Sparkles, Filter, CheckCircle2 } from 'lucide-react'

// Module-level target cache for instant 0ms switching across tabs
let cachedTargets = null

function TargetCatalog({ onSelectTarget, currentTargetId }) {
  const [allTargets, setAllTargets] = useState(cachedTargets || [])
  const [loading, setLoading] = useState(!cachedTargets)
  const [error, setError] = useState(null)

  // Filters State
  const [searchQuery, setSearchQuery] = useState('')
  const [filterClass, setFilterClass] = useState('all') // 'all', 'confirmed_planet', 'eclipsing_binary', 'false_positive'
  const [minSnr, setMinSnr] = useState(0)
  const [minConf, setMinConf] = useState(0)
  const [onlyDetrended, setOnlyDetrended] = useState(false)

  // Sorting State
  const [sortBy, setSortBy] = useState('transit_snr') // 'star_id', 'bls_period', 'bls_depth', 'transit_snr', 'confidence'
  const [sortOrder, setSortOrder] = useState('desc') // 'asc', 'desc'

  // Fetch all targets on mount if not cached
  useEffect(() => {
    if (cachedTargets && cachedTargets.length > 0) {
      setAllTargets(cachedTargets)
      setLoading(false)
      return
    }

    async function loadCatalog() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch('/api/targets?page=1&page_size=200&sort_by=transit_snr&sort_order=desc')
        if (!res.ok) throw new Error('Failed to load targets from catalog')
        const data = await res.json()
        
        let collected = data.targets || []
        cachedTargets = collected
        setAllTargets(collected)
        setLoading(false)

        const total = data.total || collected.length
        const totalPages = data.total_pages || Math.ceil(total / 200)

        // Stream remaining pages smoothly in background without freezing UI or flooding backend
        if (totalPages > 1) {
          for (let p = 2; p <= totalPages; p++) {
            try {
              const nextRes = await fetch(`/api/targets?page=${p}&page_size=200&sort_by=transit_snr&sort_order=desc`)
              if (nextRes.ok) {
                const nextData = await nextRes.json()
                if (nextData.targets && nextData.targets.length > 0) {
                  collected = collected.concat(nextData.targets)
                  cachedTargets = collected
                  setAllTargets([...collected])
                }
              }
            } catch (pageErr) {
              console.warn(`Background page ${p} fetch notice:`, pageErr)
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch target catalog:", err)
        setError("Failed to connect to the target catalog API.")
      } finally {
        setLoading(false)
      }
    }
    loadCatalog()
  }, [])

  // Filtered and Sorted Targets
  const filteredTargets = useMemo(() => {
    return allTargets
      .filter((t) => {
        // Search query filter (matches TIC ID)
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase().replace('tic', '').trim()
          if (!t.star_id.toString().includes(q)) {
            return false
          }
        }

        // Class filter
        if (filterClass !== 'all' && t.predicted_label !== filterClass) {
          return false
        }

        // Min SNR filter
        if (minSnr > 0 && (t.transit_snr || 0) < minSnr) {
          return false
        }

        // Min Confidence filter
        if (minConf > 0 && (t.confidence || 0) * 100 < minConf) {
          return false
        }

        // Only detrended filter
        if (onlyDetrended && !t.has_detrended) {
          return false
        }

        return true
      })
      .sort((a, b) => {
        let valA = a[sortBy] ?? 0
        let valB = b[sortBy] ?? 0
        if (typeof valA === 'string') {
          return sortOrder === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA)
        }
        return sortOrder === 'asc' ? valA - valB : valB - valA
      })
  }, [allTargets, searchQuery, filterClass, minSnr, minConf, onlyDetrended, sortBy, sortOrder])

  // Sorting Toggle Handler
  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setSortOrder('desc')
    }
  }

  // TanStack Virtualizer for 60 FPS scrolling
  const parentRef = useRef(null)
  const rowVirtualizer = useVirtualizer({
    count: filteredTargets.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 58,
    overscan: 8
  })

  // Quick Stats
  const planetCount = useMemo(() => allTargets.filter(t => t.predicted_label === 'confirmed_planet').length, [allTargets])
  const binaryCount = useMemo(() => allTargets.filter(t => t.predicted_label === 'eclipsing_binary').length, [allTargets])
  const fpCount = useMemo(() => allTargets.filter(t => t.predicted_label === 'false_positive').length, [allTargets])

  return (
    <div className="p-6 rounded-2xl bg-transparent border border-white/15 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.3)] flex flex-col gap-4 select-none">
      {/* ─── Header & Search ─── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-[#8CE8B5] shadow-[0_0_12px_rgba(140,232,181,0.15)]">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>Target Catalog Explorer</span>
              <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-mono tracking-wider font-semibold">
                TANSTACK VIRTUAL 60FPS
              </span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono mt-0.5">
              INDEXED TARGETS: <span className="text-[#8CE8B5] font-bold">{allTargets.length}</span> &bull; MATCHING CRITERIA: <span className="text-white font-bold">{filteredTargets.length}</span>
            </p>
          </div>
        </div>

        {/* Quick Search Box */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search TIC ID (e.g. 77202722)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-9 py-2 rounded-xl bg-black/40 border border-white/15 text-xs font-mono text-white placeholder-slate-400 focus:outline-none focus:border-[#8CE8B5]/60 focus:ring-1 focus:ring-[#8CE8B5]/30 transition-all backdrop-blur-md"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* ─── Top Telemetry KPI Cards ─── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">TOTAL INDEXED</span>
          <span className="text-xl sm:text-2xl font-bold text-white font-mono mt-1 block">
            {allTargets.length || 657}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">TESS SECTOR SURVEY</span>
        </div>
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">CONFIRMED PLANETS</span>
          <span className="text-xl sm:text-2xl font-bold text-[#8CE8B5] font-mono mt-1 block">
            {planetCount}
          </span>
          <span className="text-[10px] text-[#8CE8B5]/70 font-mono mt-0.5 block">TRANSIT EXOPLANETS</span>
        </div>
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">ECLIPSING BINARIES</span>
          <span className="text-xl sm:text-2xl font-bold text-amber-400 font-mono mt-1 block">
            {binaryCount}
          </span>
          <span className="text-[10px] text-amber-500/70 font-mono mt-0.5 block">BINARY STELLAR SYSTEMS</span>
        </div>
        <div className="p-4 rounded-xl bg-transparent hover:bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all text-center backdrop-blur-md">
          <span className="text-[10px] text-slate-400 uppercase font-mono tracking-[0.18em] block">FALSE POSITIVES</span>
          <span className="text-xl sm:text-2xl font-bold text-slate-400 font-mono mt-1 block">
            {fpCount}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">STELLAR VARIABILITY</span>
        </div>
      </div>

      {/* ─── Modern Filters & Threshold Sliders ─── */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-2xl bg-black/40 border border-white/15 backdrop-blur-md">
        {/* Class Filter Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {[
            { id: 'all', label: 'All Targets', count: allTargets.length, dot: 'bg-white' },
            { id: 'confirmed_planet', label: 'Planets', count: planetCount, dot: 'bg-[#8CE8B5]' },
            { id: 'eclipsing_binary', label: 'Binaries', count: binaryCount, dot: 'bg-amber-400' },
            { id: 'false_positive', label: 'False Positives', count: fpCount, dot: 'bg-slate-400' },
          ].map((btn) => (
            <button
              key={btn.id}
              onClick={() => setFilterClass(btn.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-mono font-bold tracking-wider uppercase transition-all cursor-pointer ${
                filterClass === btn.id
                  ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
                  : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${btn.dot}`} />
              <span>{btn.label}</span>
              <span className="text-[10px] text-slate-400 font-normal">({btn.count})</span>
            </button>
          ))}
        </div>

        {/* Sliders: SNR & Confidence + Curves Only */}
        <div className="flex items-center gap-4 flex-wrap text-xs font-mono text-slate-400">
          <div className="flex items-center gap-2">
            <span>Min SNR:</span>
            <input
              type="range"
              min="0"
              max="25"
              step="1"
              value={minSnr}
              onChange={(e) => setMinSnr(parseFloat(e.target.value))}
              className="w-20 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
            />
            <span className="text-[#8CE8B5] font-bold w-6">{minSnr}σ</span>
          </div>

          <div className="flex items-center gap-2">
            <span>Min Conf:</span>
            <input
              type="range"
              min="0"
              max="90"
              step="5"
              value={minConf}
              onChange={(e) => setMinConf(parseFloat(e.target.value))}
              className="w-20 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
            />
            <span className="text-amber-400 font-bold w-8">{minConf}%</span>
          </div>

          <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
            <input
              type="checkbox"
              checked={onlyDetrended}
              onChange={(e) => setOnlyDetrended(e.target.checked)}
              className="accent-[#8CE8B5] rounded cursor-pointer"
            />
            <span>Lightcurves Only</span>
          </label>
        </div>
      </div>

      {/* ─── Virtualized Table Header ─── */}
      <div className="grid grid-cols-12 gap-3 px-5 py-3 rounded-xl bg-black/50 border border-white/15 text-[11px] font-mono text-slate-400 uppercase tracking-wider select-none backdrop-blur-md">
        <button
          onClick={() => handleSort('star_id')}
          className={`col-span-3 flex items-center gap-1.5 text-left transition-colors cursor-pointer ${
            sortBy === 'star_id' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>Target TIC ID</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
        <button
          onClick={() => handleSort('predicted_label')}
          className={`col-span-2 flex items-center gap-1.5 text-left transition-colors cursor-pointer ${
            sortBy === 'predicted_label' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>Prediction</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
        <button
          onClick={() => handleSort('bls_period')}
          className={`col-span-2 flex items-center gap-1.5 text-right justify-end transition-colors cursor-pointer ${
            sortBy === 'bls_period' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>Period (d)</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
        <button
          onClick={() => handleSort('bls_depth')}
          className={`col-span-2 flex items-center gap-1.5 text-right justify-end transition-colors cursor-pointer ${
            sortBy === 'bls_depth' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>Depth (ΔF/F)</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
        <button
          onClick={() => handleSort('transit_snr')}
          className={`col-span-1 flex items-center gap-1.5 text-right justify-end transition-colors cursor-pointer ${
            sortBy === 'transit_snr' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>SNR</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
        <button
          onClick={() => handleSort('confidence')}
          className={`col-span-2 flex items-center gap-1.5 text-right justify-end transition-colors cursor-pointer ${
            sortBy === 'confidence' ? 'text-[#8CE8B5] font-bold' : 'hover:text-slate-200'
          }`}
        >
          <span>Confidence</span>
          <ArrowUpDown className="w-3 h-3" />
        </button>
      </div>

      {/* ─── Virtualized Scroll View (60 FPS) ─── */}
      {loading ? (
        <div className="h-[460px] rounded-2xl bg-black/60 border border-white/15 backdrop-blur-md flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-[#8CE8B5] border-t-transparent animate-spin" />
          <span className="font-mono text-xs text-slate-400">INDEXING ALL 657 TARGETS...</span>
        </div>
      ) : error ? (
        <div className="h-[460px] rounded-2xl bg-black/60 border border-rose-900/50 flex flex-col items-center justify-center gap-3">
          <span className="font-mono text-sm text-rose-400 font-bold">ERROR LOADING TARGET CATALOG</span>
          <span className="font-mono text-xs text-slate-400">{error}</span>
        </div>
      ) : filteredTargets.length === 0 ? (
        <div className="h-[460px] rounded-2xl bg-black/60 border border-white/15 flex flex-col items-center justify-center gap-2 text-slate-400 font-mono text-xs">
          <span>NO TARGETS MATCH ACTIVE FILTERS</span>
          <button
            onClick={() => {
              setSearchQuery('')
              setFilterClass('all')
              setMinSnr(0)
              setMinConf(0)
              setOnlyDetrended(false)
            }}
            className="text-[#8CE8B5] hover:underline mt-1 cursor-pointer font-bold"
          >
            Reset All Filters
          </button>
        </div>
      ) : (
        <div
          ref={parentRef}
          className="h-[460px] overflow-y-auto rounded-2xl bg-black/40 border border-white/15 custom-scrollbar pr-1 backdrop-blur-md"
        >
          <div
            style={{
              height: `${rowVirtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative'
            }}
          >
            {rowVirtualizer.getVirtualItems().map((virtualRow) => {
              const target = filteredTargets[virtualRow.index]
              if (!target) return null
              const isSelected = target.star_id === currentTargetId

              return (
                <div
                  key={target.star_id}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`
                  }}
                  className="px-1.5 py-1"
                >
                  <button
                    onClick={() => onSelectTarget && onSelectTarget(target)}
                    className={`group w-full h-full px-4 rounded-xl grid grid-cols-12 gap-3 items-center text-xs font-mono transition-all border cursor-pointer ${
                      isSelected
                        ? 'bg-white/[0.08] border-[#8CE8B5]/60 shadow-[0_0_18px_rgba(140,232,181,0.2)]'
                        : 'bg-transparent hover:bg-white/[0.04] border-white/10 hover:border-white/20'
                    }`}
                  >
                    {/* Target TIC ID */}
                    <div className="col-span-3 flex items-center gap-2.5 text-left">
                      <div
                        className={`w-2 h-2 rounded-full transition-all ${
                          isSelected
                            ? 'bg-[#8CE8B5] shadow-[0_0_8px_#8CE8B5]'
                            : target.predicted_label === 'confirmed_planet'
                            ? 'bg-[#8CE8B5]/80'
                            : target.predicted_label === 'eclipsing_binary'
                            ? 'bg-amber-400/80'
                            : 'bg-slate-600'
                        }`}
                      />
                      <span className="font-bold text-white group-hover:text-[#8CE8B5] transition-colors">
                        {target.tic_id}
                      </span>
                      {target.has_detrended && (
                        <span
                          className="text-[9px] px-1.5 py-0.5 rounded bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-semibold"
                          title="Detrended Lightcurve Available"
                        >
                          LC
                        </span>
                      )}
                    </div>

                    {/* Predicted Class Badge */}
                    <div className="col-span-2 text-left">
                      <span
                        className={`text-[10px] px-2.5 py-0.5 rounded-full font-bold tracking-wider uppercase border ${
                          target.predicted_label === 'confirmed_planet'
                            ? 'border-[#8CE8B5]/40 text-[#8CE8B5] bg-transparent'
                            : target.predicted_label === 'eclipsing_binary'
                            ? 'border-amber-500/40 text-amber-400 bg-transparent'
                            : 'border-white/10 text-slate-400 bg-transparent'
                        }`}
                      >
                        {target.predicted_label === 'confirmed_planet'
                          ? 'PLANET'
                          : target.predicted_label === 'eclipsing_binary'
                          ? 'BINARY'
                          : 'FP'}
                      </span>
                    </div>

                    {/* Period */}
                    <div className="col-span-2 text-right text-slate-200">
                      {target.bls_period ? `${target.bls_period.toFixed(3)} d` : '--'}
                    </div>

                    {/* Transit Depth */}
                    <div className="col-span-2 text-right text-[#8CE8B5] font-semibold">
                      {target.bls_depth ? `${(target.bls_depth * 100).toFixed(3)}%` : '--'}
                    </div>

                    {/* Transit SNR */}
                    <div
                      className={`col-span-1 text-right font-bold ${
                        (target.transit_snr || 0) >= 5.0 ? 'text-[#8CE8B5]' : 'text-rose-400'
                      }`}
                    >
                      {target.transit_snr ? `${target.transit_snr.toFixed(1)}σ` : '--'}
                    </div>

                    {/* Confidence Score Bar */}
                    <div className="col-span-2 flex items-center justify-end gap-2.5 text-right">
                      <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden hidden sm:block">
                        <div
                          className="h-full bg-gradient-to-r from-emerald-500 to-[#8CE8B5] rounded-full shadow-[0_0_8px_rgba(140,232,181,0.4)]"
                          style={{ width: `${(target.confidence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="font-bold text-[#8CE8B5]">
                        {target.confidence ? `${(target.confidence * 100).toFixed(0)}%` : '--'}
                      </span>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-[#8CE8B5] group-hover:translate-x-1 transition-all" />
                    </div>
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ─── Footer Telemetry Status ─── */}
      <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-2 pt-1 border-t border-white/10">
        <span>SHOWING {filteredTargets.length} OF {allTargets.length} TARGETS</span>
        <span className="text-[#8CE8B5]">CLICK ANY ROW TO MOUNT IN 3D CELESTIAL CHAMBER &rarr;</span>
      </div>
    </div>
  )
}

export default React.memo(TargetCatalog)
