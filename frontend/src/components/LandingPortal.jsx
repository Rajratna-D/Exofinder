import React, { useRef, useMemo, useEffect, useState } from 'react'
import * as THREE from 'three'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Stars } from '@react-three/drei'
import {
  Sparkles,
  ArrowUpRight,
  Target,
  Globe2,
  Database,
  ChevronRight
} from 'lucide-react'

// ─── Module-Level Caches (Zero Frame Drops on Portal Mount) ───
let cachedStarfieldPositions = null
let cachedStarfieldColors = null
let cachedStarTexture = null

// Glowing circular star sprite texture for luminous, brilliant starlight
function getStarTexture() {
  if (cachedStarTexture) return cachedStarTexture
  const c = document.createElement('canvas')
  c.width = 64
  c.height = 64
  const ctx = c.getContext('2d')
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32)
  grad.addColorStop(0, 'rgba(255, 255, 255, 1.0)')
  grad.addColorStop(0.18, 'rgba(255, 255, 255, 0.95)')
  grad.addColorStop(0.42, 'rgba(220, 245, 255, 0.70)')
  grad.addColorStop(0.72, 'rgba(140, 232, 181, 0.28)')
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0)')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 64, 64)
  cachedStarTexture = new THREE.CanvasTexture(c)
  return cachedStarTexture
}

function getStarfieldData() {
  if (cachedStarfieldPositions && cachedStarfieldColors) {
    return [cachedStarfieldPositions, cachedStarfieldColors]
  }
  const count = 2400
  const pos = new Float32Array(count * 3)
  const col = new Float32Array(count * 3)

  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 52
    pos[i * 3 + 1] = (Math.random() - 0.5) * 32
    pos[i * 3 + 2] = -Math.random() * 32 - 1.5

    const type = Math.random()
    if (type > 0.82) {
      // Radiant Cyan Starlight
      col[i * 3] = 0.82
      col[i * 3 + 1] = 0.94
      col[i * 3 + 2] = 1.0
    } else if (type > 0.70) {
      // Warm Solar Gold Starlight
      col[i * 3] = 1.0
      col[i * 3 + 1] = 0.96
      col[i * 3 + 2] = 0.84
    } else {
      // Brilliant Pure White
      col[i * 3] = 1.0
      col[i * 3 + 1] = 1.0
      col[i * 3 + 2] = 1.0
    }
  }
  cachedStarfieldPositions = pos
  cachedStarfieldColors = col
  return [pos, col]
}

let cachedHighlightPositions = null
function getHighlightStarData() {
  if (cachedHighlightPositions) return cachedHighlightPositions
  const count = 140
  const pos = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * 44
    pos[i * 3 + 1] = (Math.random() - 0.5) * 26
    pos[i * 3 + 2] = -Math.random() * 20 - 1.0
  }
  cachedHighlightPositions = pos
  return cachedHighlightPositions
}

// Prominent twinkling foreground stars
function TwinklingStarHighlights() {
  const pointsRef = useRef()
  const matRef = useRef()
  const positions = useMemo(() => getHighlightStarData(), [])
  const starTex = useMemo(() => getStarTexture(), [])

  useFrame((state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.003
    }
    if (matRef.current) {
      matRef.current.size = 0.17 + Math.sin(state.clock.elapsedTime * 2.2) * 0.035
    }
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        ref={matRef}
        size={0.17}
        map={starTex}
        color="#FFFFFF"
        transparent
        opacity={1.0}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  )
}

let cachedEarthTexture = null
function getEarthTexture() {
  if (cachedEarthTexture) return cachedEarthTexture
  const loader = new THREE.TextureLoader()
  const tex = loader.load('/assets/exoplanet_equirectangular.jpg')
  tex.colorSpace = THREE.SRGBColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  cachedEarthTexture = tex
  return cachedEarthTexture
}

// ─── Dense 3D Cosmic Starfield (Luminous, Glowing, High-Brightness Stars) ───
function DenseFrustumStarfield() {
  const pointsRef = useRef()
  const [positions, colors] = useMemo(() => getStarfieldData(), [])
  const starTex = useMemo(() => getStarTexture(), [])

  useFrame((state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.004
    }
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.105}
        map={starTex}
        vertexColors
        transparent
        opacity={1.0}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  )
}

let cachedOrbitLineGeo = null
function getOrbitLineGeometry() {
  if (cachedOrbitLineGeo) return cachedOrbitLineGeo
  const pts = []
  const a = 4.25
  const b = 2.85
  const segments = 80
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2
    pts.push(new THREE.Vector3(Math.cos(theta) * a, Math.sin(theta) * 0.85, Math.sin(theta) * b))
  }
  cachedOrbitLineGeo = new THREE.BufferGeometry().setFromPoints(pts)
  return cachedOrbitLineGeo
}

let cachedGlintTexture = null

function getHorizonGlintTexture() {
  if (cachedGlintTexture) return cachedGlintTexture
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 128
  const ctx = canvas.getContext('2d')

  // Horizontal anamorphic streak (sharp luminous beam)
  const gradH = ctx.createLinearGradient(0, 64, 512, 64)
  gradH.addColorStop(0.0, 'rgba(140, 232, 181, 0.0)')
  gradH.addColorStop(0.28, 'rgba(140, 232, 181, 0.22)')
  gradH.addColorStop(0.46, 'rgba(255, 255, 255, 0.88)')
  gradH.addColorStop(0.5, 'rgba(255, 255, 255, 1.0)') // pure white center
  gradH.addColorStop(0.54, 'rgba(255, 255, 255, 0.88)')
  gradH.addColorStop(0.72, 'rgba(140, 232, 181, 0.22)')
  gradH.addColorStop(1.0, 'rgba(140, 232, 181, 0.0)')

  ctx.fillStyle = gradH
  ctx.beginPath()
  ctx.ellipse(256, 64, 252, 5, 0, 0, Math.PI * 2)
  ctx.fill()

  // Soft radial core bloom
  const gradR = ctx.createRadialGradient(256, 64, 0, 256, 64, 46)
  gradR.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)')
  gradR.addColorStop(0.22, 'rgba(140, 232, 181, 0.75)')
  gradR.addColorStop(0.52, 'rgba(56, 189, 248, 0.25)')
  gradR.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)')

  ctx.fillStyle = gradR
  ctx.beginPath()
  ctx.arc(256, 64, 46, 0, Math.PI * 2)
  ctx.fill()

  cachedGlintTexture = new THREE.CanvasTexture(canvas)
  return cachedGlintTexture
}

// ─── 3D Celestial Space Scene (Option 2: Planetary Continuity Zoom) ───
function CelestialSpaceScene({ isLaunching = false }) {
  const { viewport } = useThree()
  const planetRef = useRef()
  const moonRef = useRef()
  const groupRef = useRef()
  const glintRef = useRef()
  const launchTimer = useRef(0)

  // Load equirectangular world map texture (Instant from Module Cache)
  const texture = useMemo(() => getEarthTexture(), [])

  // Keplerian elliptical orbit trajectory line (Precomputed Once at Startup)
  const orbitLineGeo = useMemo(() => getOrbitLineGeometry(), [])

  // Anamorphic optical horizon glint texture
  const glintTexture = useMemo(() => getHorizonGlintTexture(), [])

  // Responsive position based on screen width
  const isCompact = viewport.width < 9.5
  const targetPosX = isCompact ? 0 : Math.min(viewport.width * 0.26, 3.8)
  const targetPosY = isCompact ? 1.8 : 0.95
  const targetPosZ = isCompact ? -2.5 : -1.2

  const camPosRef = useRef(new THREE.Vector3(0, 0, 10))
  const camLookRef = useRef(new THREE.Vector3(0, 0, 0))

  useFrame((state, delta) => {
    // 1. Natural West-to-East planetary rotation on its polar axis
    if (planetRef.current) {
      planetRef.current.rotation.y += delta * 0.075
    }

    // 2. Physical 3D Keplerian orbit for the small planet / satellite moon
    if (moonRef.current) {
      const t = state.clock.getElapsedTime() * 0.36
      const a = 4.25
      const b = 2.85
      // Orbit in the inclined plane around the planet (passes both in front and behind)
      moonRef.current.position.x = Math.cos(t) * a
      moonRef.current.position.z = Math.sin(t) * b
      moonRef.current.position.y = Math.sin(t) * 0.85
      moonRef.current.rotation.y += delta * 0.18
    }

    // 3. Option 2 + 3: Planetary Continuity Zoom & Solar Horizon Glint
    if (isLaunching) {
      launchTimer.current += delta

      // Zoom camera smoothly into close orbit around the upper-right planet
      const destX = targetPosX * 0.65
      const destY = targetPosY * 0.65
      const destZ = targetPosZ + 4.2
      camPosRef.current.lerp(new THREE.Vector3(destX, destY, destZ), delta * 4.2)
      camLookRef.current.lerp(new THREE.Vector3(targetPosX, targetPosY, targetPosZ), delta * 4.2)

      // Option 3: Solar Horizon Glint (Planetary Limb Flare) — peaks between 0.18s and 0.52s
      if (glintRef.current) {
        if (launchTimer.current > 0.18 && launchTimer.current < 0.54) {
          const t = (launchTimer.current - 0.18) / 0.36 // 0.0 to 1.0
          const intensity = Math.sin(t * Math.PI) // smooth bell curve 0 -> 1 -> 0
          const scaleX = 4.0 + intensity * 5.0   // anamorphic horizontal expansion
          const scaleY = 0.45 + intensity * 0.55
          glintRef.current.scale.set(scaleX, scaleY, 1)
          glintRef.current.material.opacity = Math.pow(intensity, 1.4) * 0.95
        } else {
          glintRef.current.material.opacity = 0
        }
      }
    } else {
      launchTimer.current = 0
      if (glintRef.current) {
        glintRef.current.material.opacity = 0
      }
      // Wide panoramic view with gentle interactive mouse parallax
      const targetParallaxX = state.pointer.x * 0.3
      const targetParallaxY = state.pointer.y * 0.18
      camPosRef.current.lerp(new THREE.Vector3(targetParallaxX, targetParallaxY, 10), delta * 3.2)
      camLookRef.current.lerp(new THREE.Vector3(0, 0, 0), delta * 3.2)
    }

    state.camera.position.copy(camPosRef.current)
    state.camera.lookAt(camLookRef.current)

    // Base position for the planetary system
    if (groupRef.current) {
      groupRef.current.position.set(targetPosX, targetPosY, targetPosZ)
    }
  })

  return (
    <>
      {/* ─── Ambient Cosmic Lighting ─── */}
      <ambientLight intensity={0.18} />

      {/* Primary Star / Sunlight illuminating the day side from top-left */}
      <directionalLight position={[-8, 6, 7]} intensity={3.0} color="#FFFFFF" />

      {/* Gentle deep space blue bounce light for dark hemisphere */}
      <directionalLight position={[9, -4, -6]} intensity={0.18} color="#38BDF8" />

      {/* ─── Full-Bleed Deep Space Stars Across Camera Frustum ─── */}
      <DenseFrustumStarfield />
      <TwinklingStarHighlights />
      <Stars radius={130} depth={70} count={1600} factor={6.5} saturation={0.3} speed={0.35} />

      {/* ─── Distant Deep-Space Exoplanets (Depth & Cosmic Immersion) ─── */}
      <DistantBackgroundExoplanets />

      {/* ─── Planetary System Group (Placed in Upper-Right Quadrant) ─── */}
      <group ref={groupRef} position={[targetPosX, targetPosY, targetPosZ]}>
        {/* Main Earth-like Exoplanet with natural axial tilt */}
        <group rotation={[0.22, 0, -0.15]}>
          <mesh ref={planetRef}>
            <sphereGeometry args={[2.35, 48, 48]} />
            <meshStandardMaterial
              map={texture}
              roughness={0.65}
              metalness={0.08}
            />
          </mesh>
        </group>

        {/* Option 3: Solar Horizon Limb Glint (Anamorphic Optical Lens Flare) */}
        <sprite ref={glintRef} position={[-1.45, 1.35, 1.15]} scale={[0, 0, 1]}>
          <spriteMaterial
            map={glintTexture}
            blending={THREE.AdditiveBlending}
            transparent
            depthWrite={false}
            opacity={0}
          />
        </sprite>

        {/* Keplerian Orbit Track Line (Crisp, thin, subtle celestial white/silver) */}
        <line geometry={orbitLineGeo}>
          <lineBasicMaterial color="#94A3B8" opacity={0.30} transparent linewidth={1} />
        </line>

        {/* Small Exomoon physically orbiting in 3D around the planet */}
        <mesh ref={moonRef}>
          <sphereGeometry args={[0.36, 32, 32]} />
          <meshStandardMaterial
            color="#A0AEC0"
            roughness={0.85}
            metalness={0.12}
          />
        </mesh>
      </group>
    </>
  )
}

// Pre-cached procedural textures for distant planets
let cachedRingedTex = null
let cachedDesertTex = null

function getRingedTexture() {
  if (cachedRingedTex) return cachedRingedTex
  const c = document.createElement('canvas')
  c.width = 128
  c.height = 64
  const ctx = c.getContext('2d')
  const grad = ctx.createLinearGradient(0, 0, 0, 64)
  grad.addColorStop(0, '#1E1B4B')
  grad.addColorStop(0.25, '#312E81')
  grad.addColorStop(0.45, '#4338CA')
  grad.addColorStop(0.65, '#3B0764')
  grad.addColorStop(0.85, '#1E1B4B')
  grad.addColorStop(1.0, '#0F172A')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 128, 64)
  ctx.fillStyle = 'rgba(255,255,255,0.08)'
  ctx.fillRect(0, 20, 128, 6)
  ctx.fillRect(0, 42, 128, 4)
  cachedRingedTex = new THREE.CanvasTexture(c)
  return cachedRingedTex
}

function getDesertTexture() {
  if (cachedDesertTex) return cachedDesertTex
  const c = document.createElement('canvas')
  c.width = 128
  c.height = 64
  const ctx = c.getContext('2d')
  const grad = ctx.createLinearGradient(0, 0, 0, 64)
  grad.addColorStop(0, '#7C2D12')
  grad.addColorStop(0.35, '#C2410C')
  grad.addColorStop(0.6, '#EA580C')
  grad.addColorStop(0.85, '#9A3412')
  grad.addColorStop(1.0, '#431407')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 128, 64)
  cachedDesertTex = new THREE.CanvasTexture(c)
  return cachedDesertTex
}

// ─── Distant Deep-Space Exoplanets (Atmospheric Cosmic Depth) ───
function DistantBackgroundExoplanets() {
  const ringedRef = useRef()
  const desertRef = useRef()
  const iceRef = useRef()
  const moltenRef = useRef()
  const groupRef = useRef()

  const ringedTex = useMemo(() => getRingedTexture(), [])
  const desertTex = useMemo(() => getDesertTexture(), [])

  useFrame((state, delta) => {
    if (ringedRef.current) ringedRef.current.rotation.y += delta * 0.03
    if (desertRef.current) desertRef.current.rotation.y += delta * 0.05
    if (iceRef.current) iceRef.current.rotation.y -= delta * 0.04
    if (moltenRef.current) moltenRef.current.rotation.y += delta * 0.06

    // Subtle deep background parallax
    if (groupRef.current) {
      groupRef.current.position.x = THREE.MathUtils.lerp(groupRef.current.position.x, state.pointer.x * 0.12, 0.03)
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, state.pointer.y * 0.08, 0.03)
    }
  })

  return (
    <group ref={groupRef}>
      {/* 1. Distant Ringed Gas Giant (High Top-Left Cosmic Corner) */}
      <group position={[-18.5, 9.2, -14.0]} rotation={[0.4, 0, -0.3]}>
        <mesh ref={ringedRef}>
          <sphereGeometry args={[0.75, 32, 32]} />
          <meshStandardMaterial map={ringedTex} roughness={0.6} metalness={0.1} />
        </mesh>
        {/* Planetary Ring */}
        <mesh rotation={[-Math.PI / 2.8, 0.2, 0]}>
          <ringGeometry args={[1.05, 1.8, 48]} />
          <meshStandardMaterial
            color="#A5B4FC"
            opacity={0.38}
            transparent
            side={THREE.DoubleSide}
            roughness={0.8}
          />
        </mesh>
      </group>

      {/* 2. Distant Desert Super-Earth (Far Right Cosmic Void) */}
      <group position={[13.5, -3.2, -12.0]} rotation={[0.15, 0, 0.2]}>
        <mesh ref={desertRef}>
          <sphereGeometry args={[0.52, 32, 32]} />
          <meshStandardMaterial map={desertTex} roughness={0.8} metalness={0.1} />
        </mesh>
      </group>

      {/* 3. Distant Ocean/Ice Dwarf (Far Left Peripheral Void) */}
      <group position={[-19.0, -3.8, -15.0]}>
        <mesh ref={iceRef}>
          <sphereGeometry args={[0.42, 32, 32]} />
          <meshStandardMaterial color="#0284C7" roughness={0.35} metalness={0.15} />
        </mesh>
      </group>

      {/* 4. Tiny Distant Golden Hot Jupiter (Upper Mid-Sky Jewel) */}
      <group position={[2.5, 9.2, -18.0]}>
        <mesh ref={moltenRef}>
          <sphereGeometry args={[0.26, 24, 24]} />
          <meshStandardMaterial color="#F59E0B" roughness={0.5} metalness={0.2} />
        </mesh>
      </group>
    </group>
  )
}

// ─── Radar Scan Scope Canvas ───
function RadarScanCanvas() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let animId
    let angle = 0

    const blips = [
      { r: 0.35, theta: 0.8, size: 3.5 },
      { r: 0.65, theta: 2.4, size: 4.0 },
      { r: 0.82, theta: 4.1, size: 3.0 },
      { r: 0.50, theta: 5.2, size: 4.5 }
    ]

    const render = () => {
      const w = canvas.width
      const h = canvas.height
      const cx = w / 2
      const cy = h / 2
      const radius = Math.min(cx, cy) - 6

      ctx.clearRect(0, 0, w, h)

      // Concentric rings
      ctx.strokeStyle = 'rgba(140, 232, 181, 0.25)'
      ctx.lineWidth = 1
      for (let i = 1; i <= 3; i++) {
        ctx.beginPath()
        ctx.arc(cx, cy, (radius * i) / 3, 0, Math.PI * 2)
        ctx.stroke()
      }

      // Crosshairs
      ctx.beginPath()
      ctx.moveTo(cx - radius, cy)
      ctx.lineTo(cx + radius, cy)
      ctx.moveTo(cx, cy - radius)
      ctx.lineTo(cx + radius, cy)
      ctx.stroke()

      // Rotating sweep cone
      const sweepGradient = ctx.createConicGradient(angle, cx, cy)
      sweepGradient.addColorStop(0, 'rgba(140, 232, 181, 0.45)')
      sweepGradient.addColorStop(0.12, 'rgba(140, 232, 181, 0.0)')
      sweepGradient.addColorStop(1, 'rgba(140, 232, 181, 0.0)')

      ctx.fillStyle = sweepGradient
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.fill()

      // Target blips
      blips.forEach((b) => {
        const bx = cx + Math.cos(b.theta) * (radius * b.r)
        const by = cy + Math.sin(b.theta) * (radius * b.r)

        const diff = (angle - b.theta + Math.PI * 2) % (Math.PI * 2)
        const alpha = Math.max(0.2, 1.0 - diff / (Math.PI * 1.2))

        ctx.fillStyle = `rgba(140, 232, 181, ${alpha})`
        ctx.beginPath()
        ctx.arc(bx, by, b.size, 0, Math.PI * 2)
        ctx.fill()

        ctx.strokeStyle = `rgba(140, 232, 181, ${alpha * 0.6})`
        ctx.stroke()
      })

      angle = (angle + 0.035) % (Math.PI * 2)
      animId = requestAnimationFrame(render)
    }

    render()
    return () => cancelAnimationFrame(animId)
  }, [])

  return <canvas ref={canvasRef} width={130} height={130} className="w-24 h-24 block mx-auto" />
}

// ─── Main Landing Portal Component (Option 2: Planetary Continuity Transition) ───
function LandingPortal({
  isPortalActive = true,
  onEnterDashboard,
  onOpenCatalog,
  targetCount = 657
}) {
  const [isLaunching, setIsLaunching] = useState(false)

  // Reset transition zoom when user returns to portal
  useEffect(() => {
    if (isPortalActive) {
      setIsLaunching(false)
    }
  }, [isPortalActive])

  const handleLaunch = (tab = 'chamber') => {
    if (isLaunching) return
    setIsLaunching(true)
    setTimeout(() => {
      onEnterDashboard?.(tab)
    }, 600)
  }

  const handleOpenCatalog = () => {
    if (isLaunching) return
    setIsLaunching(true)
    setTimeout(() => {
      onOpenCatalog?.()
    }, 600)
  }

  return (
    <div className="relative w-screen h-screen min-h-screen bg-[#04060A] text-slate-100 flex flex-col justify-between overflow-hidden select-none">
      {/* ─── Full-Bleed 3D Space Background (100% Edge-to-Edge Canvas) ─── */}
      <div className="absolute inset-0 w-full h-full pointer-events-none z-0">
        <Canvas
          frameloop={isPortalActive ? 'always' : 'never'}
          camera={{ position: [0, 0, 10], fov: 45 }}
          dpr={[1, 1.25]}
          gl={{ powerPreference: 'high-performance', stencil: false }}
        >
          <CelestialSpaceScene isLaunching={isLaunching} />
        </Canvas>
      </div>

      {/* ─── Solar Horizon Lens Glint Sweep (Option 3) ─── */}
      <div
        className={`absolute pointer-events-none z-30 transition-all duration-300 ease-out ${
          isLaunching ? 'opacity-100' : 'opacity-0'
        }`}
        style={{
          top: '34%',
          right: '20%',
          transform: 'translate(50%, -50%)',
          willChange: 'opacity, transform'
        }}
      >
        <div className="relative flex items-center justify-center">
          {/* Luminous horizontal needle flare line */}
          <div
            className={`h-[2px] rounded-full bg-gradient-to-r from-transparent via-[#8CE8B5] via-white to-transparent shadow-[0_0_25px_rgba(140,232,181,0.9)] transition-all duration-500 ease-out ${
              isLaunching ? 'w-[280px] sm:w-[440px] scale-x-100 opacity-90' : 'w-0 scale-x-0 opacity-0'
            }`}
          />
          {/* Optical diamond flare center */}
          <div
            className={`absolute w-3 h-3 bg-white rotate-45 rounded-sm shadow-[0_0_20px_#8CE8B5] transition-all duration-500 delay-75 ${
              isLaunching ? 'scale-100 opacity-95' : 'scale-0 opacity-0'
            }`}
          />
        </div>
      </div>

      {/* ─── Orbit Target HUD Overlay Badge (Top Right) ─── */}
      <div className={`absolute top-7 right-12 hidden lg:flex items-center gap-2.5 px-4 py-2 rounded-xl bg-transparent border border-white/20 backdrop-blur-[2px] z-10 font-['JetBrains_Mono',monospace] text-xs transition-all duration-500 ease-out ${
        isLaunching ? 'opacity-0 -translate-y-6 pointer-events-none' : 'opacity-100 translate-y-0'
      }`}>
        <div className="w-2 h-2 rounded-full bg-[#8CE8B5] animate-ping" />
        <span className="text-slate-400 font-medium">TARGET:</span>
        <span className="text-[#8CE8B5] font-bold">TIC 77202722</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-300">PERIOD: 3.355d &bull; SNR: 28.4σ</span>
      </div>

      {/* ─── Main Hero Upper Section (Top-Left Content & Far-Right Telemetry) ─── */}
      <div className={`relative z-10 w-full max-w-[1560px] mx-auto px-6 sm:px-10 lg:px-12 pt-8 sm:pt-10 flex flex-col lg:flex-row justify-between items-start gap-4 flex-1 transition-all duration-500 ease-out ${
        isLaunching ? 'opacity-0 -translate-x-12 pointer-events-none' : 'opacity-100 translate-x-0'
      }`}>
        {/* Top-Left Content */}
        <div className="max-w-xl flex flex-col items-start my-auto">
          {/* System Status Pill */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-transparent border border-[#8CE8B5]/40 text-[#8CE8B5] font-['JetBrains_Mono',monospace] text-xs mb-5 shadow-[0_0_12px_rgba(140,232,181,0.12)]">
            <span className="w-2 h-2 rounded-full bg-[#8CE8B5] inline-block animate-pulse" />
            <span className="tracking-wider uppercase font-semibold">SYSTEM STATUS &bull; NOMINAL</span>
          </div>

          {/* Massive Bold Stacked Typography (Syne 900 Font with Tight Tracking) */}
          <h1 className="font-['Syne',sans-serif] font-[900] text-4xl sm:text-5xl md:text-6xl lg:text-[76px] tracking-[-0.03em] leading-[0.92] uppercase">
            <span className="text-white block drop-shadow-md">DETECT.</span>
            <span className="text-white block drop-shadow-md">CLASSIFY.</span>
            <span className="text-[#8CE8B5] block drop-shadow-[0_0_25px_rgba(140,232,181,0.35)]">DISCOVER.</span>
          </h1>

          {/* Subtext */}
          <p className="mt-4 text-xs sm:text-sm text-slate-300/90 max-w-md font-['Outfit',sans-serif] font-light leading-relaxed">
            Navigate the unknown, catalog new worlds, and expand the boundaries of human presence with autonomous exoplanet photometry.
          </p>

          {/* Action Buttons (Solid White Capsule + Glass Star Map Pill) */}
          <div className="mt-7 flex flex-wrap items-center gap-4">
            {/* Primary White Button (Exact Dribbble Style) */}
            <button
              onClick={() => handleLaunch('chamber')}
              className="group flex items-center gap-2.5 px-8 py-3.5 rounded-full bg-white hover:bg-slate-100 text-slate-950 font-bold text-xs sm:text-sm font-['JetBrains_Mono',monospace] tracking-wider uppercase transition-all shadow-[0_0_30px_rgba(255,255,255,0.3)] hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>LAUNCH MISSION</span>
              <ArrowUpRight className="w-4 h-4 text-slate-950 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
            </button>

            {/* Translucent Glass Pill */}
            <button
              onClick={handleOpenCatalog}
              className="flex items-center gap-2.5 px-7 py-3.5 rounded-full bg-transparent hover:bg-white/[0.08] border border-white/20 hover:border-white/40 text-slate-200 font-semibold text-xs sm:text-sm font-['JetBrains_Mono',monospace] tracking-wider uppercase transition-all shadow-lg"
            >
              <span>VIEW STAR MAP</span>
              <Sparkles className="w-4 h-4 text-[#8CE8B5]" />
            </button>
          </div>
        </div>

        {/* Far-Right Telemetry Strip (Real Scientific Observational Metrics) */}
        <div className="hidden lg:flex flex-col items-center gap-7 py-2 text-right">
          {/* Metric 1: Verified Stellar Systems */}
          <div className="flex flex-col items-end">
            <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-slate-400 mb-1.5">
              <Target className="w-4 h-4 text-[#8CE8B5]" />
            </div>
            <span className="text-2xl font-bold font-['JetBrains_Mono',monospace] text-white tracking-tight">853</span>
            <span className="text-[10px] font-['Space_Grotesk',sans-serif] font-bold text-slate-400 uppercase tracking-widest mt-0.5">STELLAR SYSTEMS</span>
          </div>

          <div className="w-6 h-px bg-white/15" />

          {/* Metric 2: Photometric Observations */}
          <div className="flex flex-col items-end">
            <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-slate-400 mb-1.5">
              <Database className="w-4 h-4 text-[#8CE8B5]" />
            </div>
            <span className="text-2xl font-bold font-['JetBrains_Mono',monospace] text-white tracking-tight">17.1M</span>
            <span className="text-[10px] font-['Space_Grotesk',sans-serif] font-bold text-slate-400 uppercase tracking-widest mt-0.5">CADENCES MAPPED</span>
          </div>

          <div className="w-6 h-px bg-white/15" />

          {/* Metric 3: AI Model Consensus */}
          <div className="flex flex-col items-end">
            <div className="w-9 h-9 rounded-xl bg-transparent border border-white/15 flex items-center justify-center text-slate-400 mb-1.5">
              <Globe2 className="w-4 h-4 text-[#8CE8B5]" />
            </div>
            <span className="text-2xl font-bold font-['JetBrains_Mono',monospace] text-white tracking-tight">94.2%</span>
            <span className="text-[10px] font-['Space_Grotesk',sans-serif] font-bold text-slate-400 uppercase tracking-widest mt-0.5">AI CONSENSUS</span>
          </div>
        </div>
      </div>

      {/* ─── Bottom Modular Dock (100% Completely Transparent Cards — Stars Visible Behind) ─── */}
      <div className={`relative z-20 w-full max-w-[1560px] mx-auto px-6 sm:px-10 lg:px-12 pb-6 pt-3 transition-all duration-500 ease-out ${
        isLaunching ? 'opacity-0 translate-y-12 pointer-events-none' : 'opacity-100 translate-y-0'
      }`}>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-stretch">
          {/* Card 1: FLEET STATUS (Wireframe 3D Spaceship) */}
          <div
            onClick={() => handleLaunch('chamber')}
            className="p-5 rounded-2xl bg-transparent hover:bg-white/[0.04] border border-white/15 hover:border-[#8CE8B5]/40 transition-all hover:-translate-y-1 cursor-pointer flex flex-col justify-between group"
          >
            <div className="flex items-center justify-between">
              <span className="font-['Space_Grotesk',sans-serif] font-bold text-xs uppercase tracking-[0.2em] text-slate-200">
                FLEET STATUS
              </span>
              <span className="font-['JetBrains_Mono',monospace] text-xs text-slate-400">3/5</span>
            </div>

            {/* Wireframe Spaceship Graphic Thumbnail */}
            <div className="my-2.5 rounded-xl overflow-hidden border border-white/10 bg-transparent flex items-center justify-center h-20 relative">
              <img
                src="/assets/spaceship_wireframe.jpg"
                alt="3D Wireframe Spaceship"
                className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent pointer-events-none" />
            </div>

            <div>
              <span className="font-['Space_Grotesk',sans-serif] font-bold text-sm text-white uppercase tracking-wider block">
                NSS VOYAGER
              </span>
              <div className="flex items-center justify-between mt-1.5">
                <span className="inline-flex items-center gap-1.5 font-['JetBrains_Mono',monospace] text-xs text-[#8CE8B5] font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#8CE8B5] animate-pulse" />
                  <span>READY</span>
                </span>
                <span className="font-['JetBrains_Mono',monospace] text-xs text-slate-400">82%</span>
              </div>
              <div className="w-full h-1 bg-white/15 rounded-full mt-1.5 overflow-hidden">
                <div className="w-4/5 h-full bg-[#8CE8B5] rounded-full" />
              </div>
            </div>
          </div>

          {/* Card 2: NEXT MISSION (Alien Landscape) */}
          <div
            onClick={() => handleLaunch('photometry')}
            className="p-5 rounded-2xl bg-transparent hover:bg-white/[0.04] border border-white/15 hover:border-[#8CE8B5]/40 transition-all hover:-translate-y-1 cursor-pointer flex flex-col justify-between group"
          >
            <div className="flex items-center justify-between">
              <span className="font-['Space_Grotesk',sans-serif] font-bold text-xs uppercase tracking-[0.2em] text-slate-200">
                NEXT MISSION
              </span>
              <span className="font-['JetBrains_Mono',monospace] text-xs text-slate-400">ETA 2D 14H</span>
            </div>

            {/* Alien Planet Landscape Thumbnail */}
            <div className="my-2.5 rounded-xl overflow-hidden border border-white/10 bg-transparent h-20 relative">
              <img
                src="/assets/kepler_landscape.jpg"
                alt="Kepler Landscape"
                className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent pointer-events-none" />
            </div>

            <div className="flex items-center justify-between mt-1">
              <div>
                <span className="font-['Space_Grotesk',sans-serif] font-bold text-sm text-white uppercase tracking-wider block">
                  KEPLER-186F
                </span>
                <span className="font-['Space_Grotesk',sans-serif] text-[10px] text-slate-400 uppercase tracking-wider block mt-0.5">
                  SURVEY MISSION
                </span>
              </div>
              <div className="w-8 h-8 rounded-full bg-transparent border border-white/20 flex items-center justify-center text-slate-300 group-hover:bg-white group-hover:text-slate-950 transition-colors shadow-sm">
                <ChevronRight className="w-4 h-4" />
              </div>
            </div>
          </div>

          {/* Card 3: SYSTEM SCAN (Live Radar Scope) */}
          <div
            onClick={handleOpenCatalog}
            className="p-5 rounded-2xl bg-transparent hover:bg-white/[0.04] border border-white/15 hover:border-[#8CE8B5]/40 transition-all hover:-translate-y-1 cursor-pointer flex flex-col justify-between group"
          >
            <div className="flex items-center justify-between">
              <span className="font-['Space_Grotesk',sans-serif] font-bold text-xs uppercase tracking-[0.2em] text-slate-200">
                SYSTEM SCAN
              </span>
              <span className="inline-flex items-center gap-1.5 font-['JetBrains_Mono',monospace] text-[10px] text-[#8CE8B5] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-[#8CE8B5] animate-ping" />
                <span>LIVE</span>
              </span>
            </div>

            {/* Rotating Circular Radar Scope */}
            <div className="my-1">
              <RadarScanCanvas />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <span className="font-['Space_Grotesk',sans-serif] font-bold text-sm text-white uppercase tracking-wider block">
                  7 SIGNALS
                </span>
                <span className="font-['Space_Grotesk',sans-serif] text-[10px] text-slate-400 uppercase tracking-wider block">
                  DETECTED
                </span>
              </div>
              <button className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-transparent border border-white/20 text-xs font-['JetBrains_Mono',monospace] text-[#8CE8B5] group-hover:bg-[#8CE8B5] group-hover:text-slate-950 transition-colors">
                <span>VIEW</span>
                <ArrowUpRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Card 4: MISSION TIMELINE (AI Vetting Pipeline) */}
          <div
            onClick={() => handleLaunch('diagnostics')}
            className="p-5 rounded-2xl bg-transparent hover:bg-white/[0.04] border border-white/15 hover:border-[#8CE8B5]/40 transition-all hover:-translate-y-1 cursor-pointer flex flex-col justify-between group"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-['Space_Grotesk',sans-serif] font-bold text-xs uppercase tracking-[0.2em] text-slate-200">
                MISSION TIMELINE
              </span>
              <span className="font-['JetBrains_Mono',monospace] text-xs text-slate-400">UPCOMING</span>
            </div>

            {/* Timeline Steps */}
            <div className="flex flex-col gap-2 my-auto">
              <div className="flex items-center justify-between text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#8CE8B5]" />
                  <span className="font-['Space_Grotesk',sans-serif] text-slate-300 font-medium">Engine Calibration</span>
                </div>
                <span className="font-['JetBrains_Mono',monospace] text-[9px] px-1.5 py-0.5 rounded bg-transparent text-[#8CE8B5] border border-[#8CE8B5]/40 font-semibold">
                  COMPLETE
                </span>
              </div>

              <div className="flex items-center justify-between text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#8CE8B5] animate-pulse" />
                  <span className="font-['Space_Grotesk',sans-serif] text-white font-bold">Gravity Assist</span>
                </div>
                <span className="font-['JetBrains_Mono',monospace] text-slate-400 text-[10px]">16:40</span>
              </div>

              <div className="p-2 rounded-xl bg-transparent border border-white/10 flex items-center justify-between text-[11px]">
                <div>
                  <span className="font-['Space_Grotesk',sans-serif] text-slate-200 block font-semibold">Arrival: Kepler-186F</span>
                  <span className="font-['JetBrains_Mono',monospace] text-[9px] text-slate-400">ETA 5D 14H</span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
              </div>
            </div>

            <div className="flex items-center gap-2 font-['JetBrains_Mono',monospace] text-[10px] text-slate-500 mt-2">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
              <span>Surface Survey Pending</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default React.memo(LandingPortal)
