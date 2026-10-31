import React, { useRef, useState, useMemo, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Stars } from '@react-three/drei'
import * as THREE from 'three'
import { Play, Pause, RotateCcw, Eye, Compass, Sliders, Sparkles, Activity, ShieldCheck, AlertCircle } from 'lucide-react'

// ─── Module-Level Texture Cache (0ms Mount Latency, Zero Freeze) ───
let cachedSunSurfaceTex = null
let cachedCoronaTexture = null
let cachedEarthSurfaceTex = null
let cachedEarthCloudsTex = null

function getSunSurfaceTexture() {
  if (cachedSunSurfaceTex) return cachedSunSurfaceTex

  const loader = new THREE.TextureLoader()
  const tex = loader.load('/assets/sun_surface.jpg')
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.anisotropy = 8
  tex.generateMipmaps = true
  tex.minFilter = THREE.LinearMipmapLinearFilter
  tex.magFilter = THREE.LinearFilter
  cachedSunSurfaceTex = tex
  return cachedSunSurfaceTex
}

function getEarthSurfaceTexture() {
  if (cachedEarthSurfaceTex) return cachedEarthSurfaceTex

  const loader = new THREE.TextureLoader()
  const tex = loader.load('/assets/earth_surface.jpg')
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.anisotropy = 4
  tex.generateMipmaps = true
  cachedEarthSurfaceTex = tex
  return cachedEarthSurfaceTex
}

function getEarthCloudsTexture() {
  if (cachedEarthCloudsTex) return cachedEarthCloudsTex

  const loader = new THREE.TextureLoader()
  const tex = loader.load('/assets/earth_clouds.jpg')
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.anisotropy = 4
  tex.generateMipmaps = true
  cachedEarthCloudsTex = tex
  return cachedEarthCloudsTex
}

function getCoronaTexture() {
  if (cachedCoronaTexture) return cachedCoronaTexture
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 512
  const ctx = canvas.getContext('2d')

  // Hollow-center radial gradient: radiates outwards from the solar limb into deep space
  // This keeps the solar photosphere surface 100% crisp and unobscured!
  const grad = ctx.createRadialGradient(256, 256, 80, 256, 256, 256)
  grad.addColorStop(0.0, 'rgba(255, 210, 90, 0.65)')   // intense solar limb aura
  grad.addColorStop(0.2, 'rgba(255, 150, 30, 0.40)')   // golden solar corona
  grad.addColorStop(0.45, 'rgba(240, 85, 10, 0.16)')   // warm orange solar atmosphere
  grad.addColorStop(0.75, 'rgba(190, 40, 5, 0.04)')    // faint reddish boundary
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)')         // transparent space

  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 512, 512)

  cachedCoronaTexture = new THREE.CanvasTexture(canvas)
  return cachedCoronaTexture
}

// ─── Option 3: High-Res NASA Sun Texture + Multi-layer Glow Shells (Locked 60 FPS) ───
function HostStar({ radius = 1.4 }) {
  const meshRef = useRef()
  const innerShellRef = useRef()
  const outerShellRef = useRef()
  const coronaRef = useRef()

  const sunSurfaceTex = useMemo(() => getSunSurfaceTexture(), [])
  const coronaTexture = useMemo(() => getCoronaTexture(), [])

  // Shell 1: Inner Chromosphere Rim Glow (Additive Fresnel Limb Shader)
  // Perfectly transparent in the center (viewDot = 1.0) so NASA surface details are crystal-clear
  const innerGlowMat = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        glowColor: { value: new THREE.Color('#FFE082') }
      },
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 glowColor;
        varying vec3 vNormal;
        void main() {
          float viewDot = dot(vNormal, vec3(0.0, 0.0, 1.0));
          float rim = 1.0 - max(0.0, viewDot);
          float intensity = pow(rim, 3.2) * 1.8;
          gl_FragColor = vec4(glowColor, intensity);
        }
      `,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
      side: THREE.FrontSide
    })
  }, [])

  // Shell 2: Outer Volumetric Plasma Halo (BackSide Fresnel Atmosphere)
  const outerGlowMat = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        glowColor: { value: new THREE.Color('#FF6D00') }
      },
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 glowColor;
        varying vec3 vNormal;
        void main() {
          float viewDot = dot(vNormal, vec3(0.0, 0.0, 1.0));
          float rim = 1.0 - abs(viewDot);
          float intensity = pow(rim, 2.5) * 0.85;
          gl_FragColor = vec4(glowColor, intensity);
        }
      `,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide
    })
  }, [])

  useFrame((state, delta) => {
    const elapsed = state.clock.elapsedTime

    // 1. Core Photosphere axial rotation
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.04
    }
    // 2. Inner chromosphere shell co-rotates with slight differential
    if (innerShellRef.current) {
      innerShellRef.current.rotation.y += delta * 0.045
    }
    // 3. Outer plasma shell counter-rotates slowly for dynamic turbulence
    if (outerShellRef.current) {
      outerShellRef.current.rotation.y -= delta * 0.02
      outerShellRef.current.rotation.z += delta * 0.015
    }
    // 4. Subtle, rhythmic solar corona breathing pulsation
    if (coronaRef.current) {
      const pulse = 1.0 + Math.sin(elapsed * 1.8) * 0.025
      const baseScale = radius * 3.8
      coronaRef.current.scale.set(baseScale * pulse, baseScale * pulse, 1)
    }
  })

  return (
    <group>
      {/* Layer 1: Core Solar Photosphere (Authentic NASA SDO Photosphere Map) */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 48, 48]} />
        <meshBasicMaterial map={sunSurfaceTex} />
      </mesh>

      {/* Layer 2: Chromosphere Shell (Inner White-Gold Fresnel Limb Glow) */}
      <mesh ref={innerShellRef} material={innerGlowMat}>
        <sphereGeometry args={[radius * 1.012, 40, 40]} />
      </mesh>

      {/* Layer 3: Outer Transition Plasma Shell (Volumetric Corona Atmosphere) */}
      <mesh ref={outerShellRef} material={outerGlowMat}>
        <sphereGeometry args={[radius * 1.09, 36, 36]} />
      </mesh>

      {/* Layer 4: Expansive Soft Solar Corona (Zero Dark Rings, Pure Radiance Framing the Star) */}
      <sprite ref={coronaRef} scale={[radius * 3.8, radius * 3.8, 1]}>
        <spriteMaterial
          map={coronaTexture}
          blending={THREE.AdditiveBlending}
          transparent
          depthWrite={false}
          opacity={0.82}
        />
      </sprite>

      {/* Radiant Solar Illumination Cast into the System */}
      <pointLight color="#FFFBEB" intensity={5.2} distance={60} decay={2} />
    </group>
  )
}

// ─── High-Performance Orbiting Earth-like Exoplanet (0ms React Stutter) ───
function Exoplanet({
  semiMajorAxis = 5.8,
  inclination = 88.5,
  radiusRatio = 0.08,
  starRadius = 1.4,
  period = 3.5,
  isPlaying = true,
  speed = 1.0,
  uiRefs
}) {
  const planetRef = useRef()
  const surfaceRef = useRef()
  const cloudsRef = useRef()
  const phaseRef = useRef(0)
  const prevTransitingRef = useRef(false)

  const earthSurfaceTex = useMemo(() => getEarthSurfaceTexture(), [])
  const earthCloudsTex = useMemo(() => getEarthCloudsTexture(), [])

  // Planet physical radius in 3D units (scaled for clear planetary fidelity)
  const planetRadius = Math.max(0.18, starRadius * radiusRatio * 2.2)

  // Orbital plane tilt in radians
  const incRad = (inclination * Math.PI) / 180.0

  // Precomputed orbit trajectory line points
  const orbitPoints = useMemo(() => {
    const pts = []
    const segments = 100
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2
      const x = Math.cos(theta) * semiMajorAxis
      const y = Math.sin(theta) * semiMajorAxis * Math.cos(incRad)
      const z = Math.sin(theta) * semiMajorAxis * Math.sin(incRad)
      pts.push(new THREE.Vector3(x, y, z))
    }
    return pts
  }, [semiMajorAxis, incRad])

  const orbitGeometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(orbitPoints)
  }, [orbitPoints])

  const frameCountRef = useRef(0)

  useFrame((state, delta) => {
    if (isPlaying) {
      // Dynamic, lively Keplerian orbit speed: ~4.5s per full orbit at 1.0x for 3.35d planet
      const basePeriodInSec = 4.5
      const periodScaled = Math.pow(Math.max(0.8, period) / 3.35, 0.42)
      const phaseDelta = (delta * speed) / (basePeriodInSec * periodScaled)
      phaseRef.current = (phaseRef.current + phaseDelta) % 1.0
    }

    const currentPhase = phaseRef.current // 0.0 to 1.0
    const centeredPhase = currentPhase > 0.5 ? currentPhase - 1.0 : currentPhase
    const theta = currentPhase * Math.PI * 2

    const x = Math.cos(theta) * semiMajorAxis
    const y = Math.sin(theta) * semiMajorAxis * Math.cos(incRad)
    const z = Math.sin(theta) * semiMajorAxis * Math.sin(incRad)

    if (planetRef.current) {
      planetRef.current.position.set(x, y, z)
    }
    // Realistic Earth axial day rotation
    if (surfaceRef.current) {
      surfaceRef.current.rotation.y += delta * 0.45
    }
    // Atmospheric winds: cloud layer drifts slightly faster over continents
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y += delta * 0.58
    }

    // Transit geometry test:
    const projectedDistance = Math.sqrt(x * x + y * y)
    const isTransiting = z > 0 && projectedDistance < (starRadius + planetRadius)

    let flux = 1.0
    if (isTransiting) {
      const overlap = Math.max(0, 1.0 - projectedDistance / starRadius)
      const depth = Math.pow(radiusRatio, 2)
      flux = Math.max(0.90, 1.0 - depth * (0.8 + 0.2 * overlap))
    }

    // ─── DIRECT DOM TELEMETRY (0ms React Stutter, Throttled to 30fps) ───
    if (uiRefs && uiRefs.current) {
      const { phaseEl, fluxEl, statusEl, tickerFluxEl, tickerCursorEl, xEl, yEl, zEl } = uiRefs.current

      // GPU cursor without layout reflow or CSS transition conflict
      if (tickerCursorEl) {
        const normX = Math.max(0, Math.min(100, ((centeredPhase + 0.15) / 0.3) * 100))
        tickerCursorEl.style.left = `${normX}%`
      }

      // Throttle text DOM writes to every 2 frames (~30fps) for 0ms frame lag
      frameCountRef.current++
      if (frameCountRef.current % 2 === 0) {
        if (phaseEl) phaseEl.textContent = centeredPhase >= 0 ? `+${centeredPhase.toFixed(4)}` : centeredPhase.toFixed(4)
        if (fluxEl) fluxEl.textContent = `${(flux * 100).toFixed(3)}%`
        if (tickerFluxEl) tickerFluxEl.textContent = `ΔF: -${((1.0 - flux) * 100).toFixed(3)}%`
        if (xEl) xEl.textContent = x.toFixed(2)
        if (yEl) yEl.textContent = y.toFixed(2)
        if (zEl) zEl.textContent = z.toFixed(2)
      }

      // Only update status badge DOM when transiting state toggles (prevents re-render lag)
      if (isTransiting !== prevTransitingRef.current) {
        prevTransitingRef.current = isTransiting
        if (statusEl) {
          if (isTransiting) {
            statusEl.className = "flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#182C22] border border-[#8CE8B5]/60 text-[#8CE8B5] font-chamber text-xs font-bold animate-pulse shadow-[0_0_15px_rgba(140,232,181,0.25)]"
            statusEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#8CE8B5] animate-ping inline-block"></span><span class="tracking-wider uppercase">TRANSIT ECLIPSE ACTIVE</span>`
          } else {
            statusEl.className = "flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-transparent border border-white/15 text-slate-400 font-chamber text-xs"
            statusEl.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-slate-500 inline-block"></span><span>OUT OF TRANSIT</span>`
          }
        }
      }
    }
  })

  return (
    <group>
      {/* Keplerian Orbital Trajectory Line */}
      <line geometry={orbitGeometry}>
        <lineBasicMaterial color="#8CE8B5" transparent opacity={0.35} linewidth={1} />
      </line>

      {/* Orbiting Earth-like Exoplanet Body */}
      <group ref={planetRef}>
        {/* 1. Earth Surface (NASA Blue Marble Continents & Deep Blue Oceans) */}
        <mesh ref={surfaceRef}>
          <sphereGeometry args={[planetRadius, 36, 36]} />
          <meshStandardMaterial
            map={earthSurfaceTex}
            roughness={0.65}
            metalness={0.05}
          />
        </mesh>

        {/* 2. Floating Cloud Atmosphere Layer (Dynamic Atmospheric Drift) */}
        <mesh ref={cloudsRef}>
          <sphereGeometry args={[planetRadius * 1.015, 36, 36]} />
          <meshStandardMaterial
            map={earthCloudsTex}
            transparent
            blending={THREE.AdditiveBlending}
            opacity={0.65}
            depthWrite={false}
          />
        </mesh>

        {/* 3. Subtle Rayleigh Atmospheric Scattering Rim Glow */}
        <mesh>
          <sphereGeometry args={[planetRadius * 1.028, 28, 28]} />
          <meshBasicMaterial
            color="#60A5FA"
            transparent
            opacity={0.22}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            side={THREE.BackSide}
          />
        </mesh>
      </group>
    </group>
  )
}

// ─── Camera Controller for Presets ───
function CameraController({ viewMode }) {
  const controlsRef = useRef()

  useEffect(() => {
    if (!controlsRef.current) return
    if (viewMode === 'transit') {
      controlsRef.current.object.position.set(0, 0, 14)
      controlsRef.current.target.set(0, 0, 0)
    } else if (viewMode === 'topdown') {
      controlsRef.current.object.position.set(0, 16, 0.1)
      controlsRef.current.target.set(0, 0, 0)
    } else if (viewMode === 'angled') {
      controlsRef.current.object.position.set(10, 7, 12)
      controlsRef.current.target.set(0, 0, 0)
    }
    controlsRef.current.update()
  }, [viewMode])

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      minDistance={4}
      maxDistance={25}
      maxPolarAngle={Math.PI / 2 + 0.1}
    />
  )
}

// ─── Main 3D Celestial Chamber Component (Smooth 60FPS & Realistic Sun-Like Star) ───
function CelestialChamber({
  starName = "TIC 77202722",
  defaultPeriod = 3.35,
  defaultDepth: _defaultDepth = 0.0135,
  defaultDuration: _defaultDuration = 0.09,
  defaultRadiusRatio = 0.095,
  defaultInclination = 88.5,
  isActive = true
}) {
  // Simulator State (Only updates on user input, NOT on 60FPS animation ticks!)
  const [isPlaying, setIsPlaying] = useState(true)
  const [simSpeed, setSimSpeed] = useState(1.0)
  const [viewMode, setViewMode] = useState('angled') // 'angled', 'transit', 'topdown'
  
  // Interactive Keplerian Controls
  const [inclination, setInclination] = useState(defaultInclination)
  const [radiusRatio, setRadiusRatio] = useState(defaultRadiusRatio)
  const [period, setPeriod] = useState(defaultPeriod)

  // Smoothly update Keplerian parameters when target changes without requiring Canvas remount
  useEffect(() => {
    setPeriod(defaultPeriod)
    setRadiusRatio(defaultRadiusRatio)
    setInclination(defaultInclination)
  }, [defaultPeriod, defaultRadiusRatio, defaultInclination])

  // Direct DOM Refs for High-Frequency Telemetry (Eliminates 60FPS React Re-Render Lag)
  const uiRefs = useRef({
    phaseEl: null,
    fluxEl: null,
    statusEl: null,
    tickerFluxEl: null,
    tickerCursorEl: null,
    xEl: null,
    yEl: null,
    zEl: null
  })

  return (
    <div className="font-chamber relative w-full h-[580px] rounded-2xl overflow-hidden bg-black/60 border border-white/15 backdrop-blur-md shadow-[0_12px_40px_rgba(0,0,0,0.6)] select-none">
      {/* ─── Top Telemetry Header HUD ─── */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-6 py-4 bg-gradient-to-b from-black/85 via-black/35 to-transparent pointer-events-none">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-transparent border border-white/15 text-[#8CE8B5] backdrop-blur-md">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <span className="font-chamber font-bold text-base tracking-wider text-white uppercase">
                {starName}
              </span>
              <span className="font-chamber font-bold text-[11px] uppercase tracking-[0.18em] px-2 py-0.5 rounded-full border border-white/15 text-slate-300">
                3D KEPLER SIM
              </span>
            </div>
            <p className="text-xs text-slate-400 font-chamber mt-0.5">
              ORBITAL PHASE Φ: <span ref={(el) => (uiRefs.current.phaseEl = el)} className="text-[#8CE8B5] font-semibold">+0.0000</span>
            </p>
          </div>
        </div>

        {/* Live Transit Alert Badge */}
        <div className="flex items-center gap-3 pointer-events-auto">
          <div
            ref={(el) => (uiRefs.current.statusEl = el)}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-transparent border border-white/15 text-slate-400 font-chamber text-xs"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-slate-500" />
            <span>OUT OF TRANSIT</span>
          </div>

          {/* Current Flux Telemetry Indicator */}
          <div className="hidden sm:flex flex-col items-end px-3.5 py-1 rounded-xl bg-transparent border border-white/15 backdrop-blur-md">
            <span className="font-chamber text-[11px] font-bold text-slate-400 uppercase tracking-wider">CURRENT FLUX</span>
            <span ref={(el) => (uiRefs.current.fluxEl = el)} className="text-xs font-bold font-chamber text-white">
              100.000%
            </span>
          </div>
        </div>
      </div>

      {/* ─── 3D WebGL Canvas Stage (Small Shining Stars ONLY inside the box) ─── */}
      <div className="w-full h-full cursor-grab active:cursor-grabbing">
        <Canvas
          frameloop={isActive ? 'always' : 'never'}
          camera={{ position: [10, 7, 12], fov: 42 }}
          dpr={[1, 1.25]}
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
            stencil: false,
            depth: true
          }}
        >
          <color attach="background" args={['#04060A']} />

          {/* Small shining stars ONLY inside the box where planet is orbiting the star */}
          <Stars radius={95} depth={55} count={900} factor={3.2} saturation={0.4} fade speed={0.2} />
          
          {/* Ambient Cosmic Lighting */}
          <ambientLight intensity={0.25} />

          {/* Realistic Sun-Like Star */}
          <HostStar radius={1.4} />

          {/* Orbiting Exoplanet with Dynamic Physics */}
          <Exoplanet
            semiMajorAxis={5.8}
            inclination={inclination}
            radiusRatio={radiusRatio}
            starRadius={1.4}
            period={period}
            isPlaying={isPlaying}
            speed={simSpeed}
            uiRefs={uiRefs}
          />

          {/* OrbitControls with Camera View Presets */}
          <CameraController viewMode={viewMode} />
        </Canvas>
      </div>

      {/* ─── Left View Presets Overlay ─── */}
      <div className="absolute top-20 left-5 z-10 flex flex-col gap-1.5 p-1.5 rounded-2xl bg-black/50 border border-white/15 backdrop-blur-md shadow-xl">
        <button
          onClick={() => setViewMode('angled')}
          title="Isometric View"
          className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-chamber font-bold uppercase tracking-wider transition-all ${
            viewMode === 'angled'
              ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
          }`}
        >
          <Compass className="w-3.5 h-3.5 text-[#8CE8B5]" />
          <span>Orbital</span>
        </button>
        <button
          onClick={() => setViewMode('transit')}
          title="Line of Sight (Transit Eclipse)"
          className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-chamber font-bold uppercase tracking-wider transition-all ${
            viewMode === 'transit'
              ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
          }`}
        >
          <Eye className="w-3.5 h-3.5 text-[#8CE8B5]" />
          <span>Transit</span>
        </button>
        <button
          onClick={() => setViewMode('topdown')}
          title="Top-Down View"
          className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-chamber font-bold uppercase tracking-wider transition-all ${
            viewMode === 'topdown'
              ? 'bg-white/10 text-white border border-[#8CE8B5]/50 shadow-[0_0_12px_rgba(140,232,181,0.2)]'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04] border border-transparent'
          }`}
        >
          <Activity className="w-3.5 h-3.5 text-[#8CE8B5]" />
          <span>Polar</span>
        </button>
      </div>

      {/* ─── Bottom-Right Keplerian Sliders Control Tray ─── */}
      <div className="absolute bottom-16 right-5 z-10 p-4 rounded-2xl bg-black/60 border border-white/15 backdrop-blur-xl w-72 space-y-3 shadow-2xl">
        <div className="flex items-center justify-between pb-1.5 border-b border-white/10">
          <span className="font-chamber font-bold text-xs uppercase tracking-[0.18em] text-slate-200 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-[#8CE8B5]" /> Parameters
          </span>
          <button
            onClick={() => {
              setInclination(defaultInclination)
              setRadiusRatio(defaultRadiusRatio)
              setPeriod(defaultPeriod)
            }}
            title="Reset to defaults"
            className="text-slate-500 hover:text-[#8CE8B5] transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>

        {/* Inclination Slider */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-chamber">
            <span className="text-slate-400">Inclination (i)</span>
            <span className="text-[#8CE8B5] font-bold">{inclination.toFixed(1)}°</span>
          </div>
          <input
            type="range"
            min="65"
            max="90"
            step="0.5"
            value={inclination}
            onChange={(e) => setInclination(parseFloat(e.target.value))}
            className="w-full h-1 bg-white/15 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-chamber">
            <span>65° (grazing)</span>
            <span>90° (central)</span>
          </div>
        </div>

        {/* Radius Ratio Rp/Rs Slider */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-chamber">
            <span className="text-slate-400">Radius Ratio (Rp/R*)</span>
            <span className="text-[#8CE8B5] font-bold">{radiusRatio.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min="0.02"
            max="0.20"
            step="0.005"
            value={radiusRatio}
            onChange={(e) => setRadiusRatio(parseFloat(e.target.value))}
            className="w-full h-1 bg-white/15 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-chamber">
            <span>Earth (0.01)</span>
            <span>Jupiter (0.10)</span>
          </div>
        </div>

        {/* Orbital Period Slider */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-chamber">
            <span className="text-slate-400">Period (P)</span>
            <span className="text-[#8CE8B5] font-bold">{period.toFixed(2)} d</span>
          </div>
          <input
            type="range"
            min="0.8"
            max="15"
            step="0.2"
            value={period}
            onChange={(e) => setPeriod(parseFloat(e.target.value))}
            className="w-full h-1 bg-white/15 rounded-lg appearance-none cursor-pointer accent-[#8CE8B5]"
          />
        </div>
      </div>

      {/* ─── Bottom Navigation & Mini Synchronized Lightcurve Ticker ─── */}
      <div className="absolute bottom-0 left-0 right-0 z-10 px-6 py-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent border-t border-white/10 flex items-center justify-between gap-4">
        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center justify-center w-8 h-8 rounded-xl bg-transparent hover:bg-white/10 border border-white/20 text-white transition-all shadow-sm"
          >
            {isPlaying ? <Pause className="w-4 h-4 text-[#8CE8B5]" /> : <Play className="w-4 h-4 ml-0.5 text-[#8CE8B5]" />}
          </button>

          {/* Speed Multiplier Pill */}
          <div className="flex items-center rounded-xl bg-transparent border border-white/15 p-0.5 backdrop-blur-md">
            {[0.5, 1.0, 2.0, 4.0].map((s) => (
              <button
                key={s}
                onClick={() => setSimSpeed(s)}
                className={`px-2.5 py-1 rounded-lg text-xs font-chamber transition-all ${
                  simSpeed === s
                    ? 'bg-white/15 text-[#8CE8B5] font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Mini Synchronized Transit Curve Ticker */}
        <div className="flex-1 max-w-sm hidden md:flex items-center gap-3 px-3.5 py-1.5 rounded-xl bg-transparent border border-white/15 backdrop-blur-md">
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400 font-chamber font-bold uppercase tracking-wider">SYNC TRANSIT</span>
            <span ref={(el) => (uiRefs.current.tickerFluxEl = el)} className="text-xs text-[#8CE8B5] font-chamber font-bold">
              ΔF: -0.000%
            </span>
          </div>

          {/* Mini Interactive SVG Lightcurve with live phase cursor */}
          <div className="relative flex-1 h-8 bg-black/40 rounded border border-white/10 overflow-hidden">
            <svg className="w-full h-full" viewBox="0 0 120 30" preserveAspectRatio="none">
              {/* Baseline reference */}
              <line x1="0" y1="6" x2="120" y2="6" stroke="#475569" strokeDasharray="2 2" strokeWidth="0.8" />
              {/* Theoretical Transit Dip Profile */}
              <path
                d="M 0 6 L 45 6 Q 60 24 75 6 L 120 6"
                fill="none"
                stroke="#8CE8B5"
                strokeWidth="1.5"
                opacity="0.85"
              />
            </svg>

            {/* Sweeping Live Phase Cursor */}
            <div
              ref={(el) => (uiRefs.current.tickerCursorEl = el)}
              className="absolute top-0 bottom-0 w-0.5 bg-[#8CE8B5] shadow-[0_0_8px_#8CE8B5] pointer-events-none will-change-[left]"
              style={{ left: '50%' }}
            />
          </div>
        </div>

        {/* Coordinate HUD */}
        <div className="hidden lg:flex items-center gap-3 text-xs font-chamber text-slate-400">
          <span>X: <b ref={(el) => (uiRefs.current.xEl = el)} className="text-white">0.00</b></span>
          <span>Y: <b ref={(el) => (uiRefs.current.yEl = el)} className="text-white">0.00</b></span>
          <span>Z: <b ref={(el) => (uiRefs.current.zEl = el)} className="text-white">0.00</b></span>
        </div>
      </div>
    </div>
  )
}

export default React.memo(CelestialChamber)
