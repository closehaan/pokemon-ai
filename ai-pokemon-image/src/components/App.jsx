import { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react'
import '../styles/App.css'

/** Fixed fallback threshold when classifying pixels (no UI control) */
const FIXED_BACKGROUND_THRESHOLD = 0.5

/** Overlay tints for mask preview (distinct from each other) */
const MASK_OVERLAY = {
  subject: { r: 236, g: 72, b: 153, a: 170, label: 'Subject', css: '#ec4899' },
  sky: { r: 14, g: 165, b: 233, a: 170, label: 'Sky', css: '#0ea5e9' },
  water: { r: 6, g: 182, b: 212, a: 170, label: 'Water', css: '#06b6d4' },
  grass: { r: 74, g: 222, b: 128, a: 170, label: 'Grass', css: '#4ade80' },
  ground: { r: 180, g: 83, b: 9, a: 170, label: 'Ground', css: '#b45309' },
  other: { r: 168, g: 85, b: 247, a: 170, label: 'Other background', css: '#a855f7' },
}

/**
 * Decode RGB from rembg_server.py create_visualization (tolerant to PNG compression).
 * Server: magenta=subject, light blue=sky, lime=grass, brown=ground, deep sky=water, gray=other
 */
function matchServerVisualizationColor(r, g, b) {
  if (r > 200 && b > 200 && g < 95) return 'subject'
  if (r >= 90 && r <= 180 && g >= 165 && g <= 245 && b >= 195 && b <= 255) return 'sky'
  if (g >= 130 && r <= 125 && b <= 140 && g > r + 28 && g > b + 22 && r >= 12) return 'grass'
  if (r >= 95 && r <= 200 && g >= 28 && g <= 125 && b <= 90 && r >= g - 20 && r > b + 25) return 'ground'
  if (r <= 115 && g >= 145 && b >= 185 && b >= g - 30) return 'water'
  if (r >= 125 && r <= 200 && g >= 125 && g <= 200 && b >= 125 && b <= 200) {
    if (Math.max(r, g, b) - Math.min(r, g, b) < 50) return 'other'
  }
  return null
}

/**
 * Pixel rect where the image is actually painted for `object-fit: contain`
 * (coordinates relative to the img element's top-left).
 */
function getImageObjectFitContainRect(img) {
  const nw = img.naturalWidth
  const nh = img.naturalHeight
  if (!nw || !nh) return null
  const w = img.clientWidth
  const h = img.clientHeight
  if (!w || !h) return null
  const scale = Math.min(w / nw, h / nh)
  const drawW = nw * scale
  const drawH = nh * scale
  const offX = (w - drawW) / 2
  const offY = (h - drawH) / 2
  return { offX, offY, drawW, drawH, stageW: w, stageH: h, nw, nh }
}

/** PokeAPI type.name values allowed for each segmentation background region */
const TYPES_FOR_BACKGROUND = {
  sky: new Set(['flying', 'dragon']),
  grass: new Set(['grass', 'bug']),
  water: new Set(['water']),
  ground: new Set(['rock', 'ground', 'fire']),
}

/**
 * Sky: Flying, Dragon / Grass: Grass, Bug / Water: Water /
 * Ground: Rock, Ground, Fire / Other: any type
 */
function pokemonMatchesBackgroundCategory(apiTypes, bgCategory) {
  if (bgCategory === 'other') return true
  const allowed = TYPES_FOR_BACKGROUND[bgCategory]
  if (!allowed) return true
  return apiTypes.some((t) => allowed.has(t.type.name))
}

function classifyFallbackPixel(r, g, b, backgroundThreshold, recalcVariant) {
  const brightness = (r + g + b) / 3
  const saturation = Math.max(r, g, b) - Math.min(r, g, b)

  const skyBright = recalcVariant
    ? 120 + (1 - backgroundThreshold) * 40
    : 90 + (1 - backgroundThreshold) * 40
  const skyBlue = recalcVariant
    ? 130 + (1 - backgroundThreshold) * 30
    : 90 + (1 - backgroundThreshold) * 30
  const groundGreen = 100 - (1 - backgroundThreshold) * 20
  const groundBright = 200 + (1 - backgroundThreshold) * 30

  const isSky = recalcVariant
    ? brightness > skyBright - 50 &&
      b > skyBlue - 50 &&
      (b >= r || b >= g) &&
      (b > r + 5 || b > g + 5) &&
      saturation < 180
    : brightness > skyBright &&
      b > skyBlue &&
      (b > r || b > g) &&
      (b > r + 10 || b > g + 10) &&
      saturation < 160

  const isGreenDominant = g > r + 20 && g > b + 20
  const isBrownish = r > 100 && g > 80 && b < 100 && Math.abs(r - g) < 30

  const isGrass =
    !isSky &&
    b < (recalcVariant ? 90 : 100) &&
    isGreenDominant &&
    g > groundGreen + (recalcVariant ? 8 : 15) &&
    brightness < groundBright &&
    saturation > (recalcVariant ? 28 : 35)

  const isGround =
    !isSky &&
    !isGrass &&
    b < (recalcVariant ? 90 : 100) &&
    ((isGreenDominant && g > groundGreen && brightness < groundBright) ||
      (isBrownish && brightness < 200) ||
      (g > 90 && r > 70 && b < 90 && brightness < 210 && b < r && b < g) ||
      (r > 80 && g > 70 && b < (recalcVariant ? 70 : 80) && brightness < 190))

  if (isSky) return 'sky'
  if (isGrass) return 'grass'
  if (isGround) return 'ground'
  return 'other'
}

function App() {
  const [baseImage, setBaseImage] = useState(null)
  const [baseImageFile, setBaseImageFile] = useState(null)
  const [backgroundMask, setBackgroundMask] = useState(null) // Combined background mask
  const [subjectMask, setSubjectMask] = useState(null) // Subject mask
  const [skyMask, setSkyMask] = useState(null) // Sky mask
  const [waterMask, setWaterMask] = useState(null) // Water mask
  const [grassMask, setGrassMask] = useState(null) // Grass mask
  const [groundMask, setGroundMask] = useState(null) // Ground (non-grass) mask
  const [otherMask, setOtherMask] = useState(null) // Other background mask
  const [originalImageData, setOriginalImageData] = useState(null) // Store original processed image data
  const [pokemonList, setPokemonList] = useState([]) // Array of {imageUrl, scale, left, top}
  const [isLoading, setIsLoading] = useState(false)
  const [isProcessingBackground, setIsProcessingBackground] = useState(false)
  /** 1–6 Pokémon; 3 (middle) = random count 1–6 */
  const [pokemonAmount, setPokemonAmount] = useState(3)
  const [showMaskOverlay, setShowMaskOverlay] = useState(false)
  const fileInputRef = useRef(null)
  const imageRef = useRef(null)
  const maskCanvasRef = useRef(null)
  /** object-fit content rect; keeps mask + Pokémon aligned to the visible photo */
  const [imageLayout, setImageLayout] = useState(null)
  
  // SAM API endpoint - defaults to localhost:5001, can be overridden with env variable
  const SAM_API_URL = import.meta.env.VITE_SAM_API_URL || 'http://localhost:5001'
  const SEGMENT_API_URL = `${SAM_API_URL}/segment`
  const REMOVE_BG_API_URL = `${SAM_API_URL}/remove-background`

  const updateImageLayout = useCallback(() => {
    const img = imageRef.current
    if (!img?.naturalWidth) {
      setImageLayout(null)
      return
    }
    setImageLayout(getImageObjectFitContainRect(img))
  }, [])

  useLayoutEffect(() => {
    if (!baseImage) {
      setImageLayout(null)
      return
    }
    updateImageLayout()
    const img = imageRef.current
    if (!img) return
    const ro = new ResizeObserver(() => updateImageLayout())
    ro.observe(img)
    const vv = window.visualViewport
    const onVv = () => updateImageLayout()
    vv?.addEventListener('resize', onVv)
    vv?.addEventListener('scroll', onVv)
    window.addEventListener('orientationchange', onVv)
    return () => {
      ro.disconnect()
      vv?.removeEventListener('resize', onVv)
      vv?.removeEventListener('scroll', onVv)
      window.removeEventListener('orientationchange', onVv)
    }
  }, [baseImage, updateImageLayout])

  const handleImageSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      const imageUrl = URL.createObjectURL(file)
      setBaseImage(imageUrl)
      setBaseImageFile(file)
      // Clear Pokemon overlay when new image is selected
      setPokemonList([])
      setBackgroundMask(null)
      setSubjectMask(null)
      setSkyMask(null)
      setWaterMask(null)
      setGrassMask(null)
      setGroundMask(null)
      setOtherMask(null)
      // Process background after image loads
      setIsProcessingBackground(true)
    }
  }

  useEffect(() => {
    const processBackground = async () => {
      if (!baseImageFile || !imageRef.current) return

      try {
        // Wait for image to load
        await new Promise((resolve, reject) => {
          if (imageRef.current.complete && imageRef.current.naturalWidth > 0) {
            resolve()
          } else {
            imageRef.current.onload = resolve
            imageRef.current.onerror = reject
            setTimeout(() => reject(new Error('Image load timeout')), 10000)
          }
        })

        // Check if image is valid
        if (!imageRef.current.naturalWidth || !imageRef.current.naturalHeight) {
          throw new Error('Invalid image dimensions')
        }

        // Call SAM segment API to get segmentation mask
        const formData = new FormData()
        formData.append('image', baseImageFile)

        const response = await fetch(SEGMENT_API_URL, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Unknown error')
          throw new Error(`API error: ${response.status} - ${errorText}`)
        }

        const blob = await response.blob()
        
        if (!blob || blob.size === 0) {
          throw new Error('Segmentation returned empty result')
        }

        // Create canvas to get mask data from the result
        // Segmentation returns colored mask: red=subject, blue=sky, green=ground, yellow=other
        const img = new Image()
        const blobUrl = URL.createObjectURL(blob)
        img.src = blobUrl
        
        await new Promise((resolve, reject) => {
          img.onload = resolve
          img.onerror = reject
          setTimeout(() => reject(new Error('Segmentation image load timeout')), 10000)
        })

        const canvas = document.createElement('canvas')
        canvas.width = img.width
        canvas.height = img.height
        const ctx = canvas.getContext('2d')
        
        if (!ctx) {
          throw new Error('Could not get canvas context')
        }
        
        ctx.drawImage(img, 0, 0)

        // Get image data from segmentation result
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        
        // Store original image data for threshold recalculation
        setOriginalImageData({
          data: new Uint8ClampedArray(imageData.data),
          width: canvas.width,
          height: canvas.height
        })

        const n = imageData.data.length / 4
        const subject = new Uint8Array(n)
        const sky = new Uint8Array(n)
        const water = new Uint8Array(n)
        const grass = new Uint8Array(n)
        const ground = new Uint8Array(n)
        const other = new Uint8Array(n)
        const combinedBackground = new Uint8Array(n)

        for (let i = 0; i < imageData.data.length; i += 4) {
          const r = imageData.data[i]
          const g = imageData.data[i + 1]
          const b = imageData.data[i + 2]
          const ix = i / 4
          const fromServer = matchServerVisualizationColor(r, g, b)
          const cat = fromServer ?? classifyFallbackPixel(r, g, b, FIXED_BACKGROUND_THRESHOLD, false)
          subject[ix] = cat === 'subject' ? 1 : 0
          sky[ix] = cat === 'sky' ? 1 : 0
          water[ix] = cat === 'water' ? 1 : 0
          grass[ix] = cat === 'grass' ? 1 : 0
          ground[ix] = cat === 'ground' ? 1 : 0
          other[ix] = cat === 'other' ? 1 : 0
          combinedBackground[ix] = cat === 'subject' ? 0 : 1
        }

        // Clean up blob URL
        URL.revokeObjectURL(blobUrl)

        setSubjectMask({
          data: subject,
          width: canvas.width,
          height: canvas.height
        })
        
        setSkyMask({
          data: sky,
          width: canvas.width,
          height: canvas.height
        })

        setWaterMask({
          data: water,
          width: canvas.width,
          height: canvas.height
        })

        setGrassMask({
          data: grass,
          width: canvas.width,
          height: canvas.height
        })

        setGroundMask({
          data: ground,
          width: canvas.width,
          height: canvas.height
        })

        setOtherMask({
          data: other,
          width: canvas.width,
          height: canvas.height
        })

        setBackgroundMask({
          data: combinedBackground,
          width: canvas.width,
          height: canvas.height
        })
      } catch (error) {
        console.error('Error processing background:', error)
        alert(`Failed to process background: ${error.message || 'Unknown error'}. Make sure your SAM backend server is running at ${SAM_API_URL}`)
        setIsProcessingBackground(false)
      } finally {
        setIsProcessingBackground(false)
      }
    }

    if (baseImageFile && imageRef.current) {
      // Always process when baseImageFile changes (new image selected)
      processBackground()
    }
  }, [baseImageFile, SEGMENT_API_URL])

  // Recalculate masks when threshold changes (without calling API)
  useEffect(() => {
    if (!originalImageData) return

    const n = originalImageData.data.length / 4
    const subject = new Uint8Array(n)
    const sky = new Uint8Array(n)
    const water = new Uint8Array(n)
    const grass = new Uint8Array(n)
    const ground = new Uint8Array(n)
    const other = new Uint8Array(n)
    const combinedBackground = new Uint8Array(n)

    for (let i = 0; i < originalImageData.data.length; i += 4) {
      const r = originalImageData.data[i]
      const g = originalImageData.data[i + 1]
      const b = originalImageData.data[i + 2]
      const ix = i / 4
      const fromServer = matchServerVisualizationColor(r, g, b)
      const cat = fromServer ?? classifyFallbackPixel(r, g, b, FIXED_BACKGROUND_THRESHOLD, true)
      subject[ix] = cat === 'subject' ? 1 : 0
      sky[ix] = cat === 'sky' ? 1 : 0
      water[ix] = cat === 'water' ? 1 : 0
      grass[ix] = cat === 'grass' ? 1 : 0
      ground[ix] = cat === 'ground' ? 1 : 0
      other[ix] = cat === 'other' ? 1 : 0
      combinedBackground[ix] = cat === 'subject' ? 0 : 1
    }

    setSubjectMask({
      data: subject,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setSkyMask({
      data: sky,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setWaterMask({
      data: water,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setGrassMask({
      data: grass,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setGroundMask({
      data: ground,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setOtherMask({
      data: other,
      width: originalImageData.width,
      height: originalImageData.height
    })

    setBackgroundMask({
      data: combinedBackground,
      width: originalImageData.width,
      height: originalImageData.height
    })
  }, [originalImageData])

  // Update mask overlay visualization (canvas matches object-fit content box only)
  useEffect(() => {
    if (
      !backgroundMask ||
      !subjectMask ||
      !skyMask ||
      !waterMask ||
      !grassMask ||
      !groundMask ||
      !otherMask ||
      !maskCanvasRef.current ||
      !showMaskOverlay ||
      !imageLayout
    )
      return

    const canvas = maskCanvasRef.current
    const ow = Math.max(1, Math.round(imageLayout.drawW))
    const oh = Math.max(1, Math.round(imageLayout.drawH))

    canvas.width = ow
    canvas.height = oh

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Map mask pixels 1:1 onto the visible image area (same as natural image aspect)
    const scaleX = ow / backgroundMask.width
    const scaleY = oh / backgroundMask.height

    // Create image data for overlay
    const imageData = ctx.createImageData(canvas.width, canvas.height)
    
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const maskX = Math.floor(x / scaleX)
        const maskY = Math.floor(y / scaleY)
        
        if (maskX >= 0 && maskX < backgroundMask.width && maskY >= 0 && maskY < backgroundMask.height) {
          const maskIndex = maskY * backgroundMask.width + maskX
          
          // Check each category
          const isSubject = subjectMask.data[maskIndex] === 1
          const isSky = skyMask.data[maskIndex] === 1
          const isWater = waterMask.data[maskIndex] === 1
          const isGrass = grassMask.data[maskIndex] === 1
          const isGround = groundMask.data[maskIndex] === 1
          const isOther = otherMask.data[maskIndex] === 1

          const pixelIndex = (y * canvas.width + x) * 4

          if (isSubject) {
            const c = MASK_OVERLAY.subject
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else if (isSky) {
            const c = MASK_OVERLAY.sky
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else if (isWater) {
            const c = MASK_OVERLAY.water
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else if (isGrass) {
            const c = MASK_OVERLAY.grass
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else if (isGround) {
            const c = MASK_OVERLAY.ground
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else if (isOther) {
            const c = MASK_OVERLAY.other
            imageData.data[pixelIndex] = c.r
            imageData.data[pixelIndex + 1] = c.g
            imageData.data[pixelIndex + 2] = c.b
            imageData.data[pixelIndex + 3] = c.a
          } else {
            // No overlay
            imageData.data[pixelIndex] = 0
            imageData.data[pixelIndex + 1] = 0
            imageData.data[pixelIndex + 2] = 0
            imageData.data[pixelIndex + 3] = 0
          }
        }
      }
    }
    
    ctx.putImageData(imageData, 0, 0)
  }, [
    backgroundMask,
    subjectMask,
    skyMask,
    waterMask,
    grassMask,
    groundMask,
    otherMask,
    showMaskOverlay,
    imageLayout,
  ])

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleBack = () => {
    if (baseImage && baseImage.startsWith('blob:')) {
      URL.revokeObjectURL(baseImage)
    }
    setBaseImage(null)
    setBaseImageFile(null)
    setPokemonList([])
    setBackgroundMask(null)
    setSubjectMask(null)
    setSkyMask(null)
    setWaterMask(null)
    setGrassMask(null)
    setGroundMask(null)
    setOtherMask(null)
    setOriginalImageData(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Check if two Pokemon rectangles overlap (with padding to prevent being too close)
  const checkCollision = (pokemon1, pokemon2, padding = 10) => {
    const pokemon1Right = pokemon1.x + pokemon1.width + padding
    const pokemon1Bottom = pokemon1.y + pokemon1.height + padding
    const pokemon1Left = pokemon1.x - padding
    const pokemon1Top = pokemon1.y - padding
    
    const pokemon2Right = pokemon2.x + pokemon2.width + padding
    const pokemon2Bottom = pokemon2.y + pokemon2.height + padding
    const pokemon2Left = pokemon2.x - padding
    const pokemon2Top = pokemon2.y - padding

    // Check if rectangles overlap
    return !(
      pokemon1Right < pokemon2Left ||
      pokemon1Left > pokemon2Right ||
      pokemon1Bottom < pokemon2Top ||
      pokemon1Top > pokemon2Bottom
    )
  }

  // Check if a position collides with existing Pokemon
  const checkCollisionWithExisting = (x, y, width, height, existingPokemon) => {
    const newPokemon = { x, y, width, height }
    
    for (const existing of existingPokemon) {
      if (checkCollision(newPokemon, existing)) {
        return true
      }
    }
    return false
  }

  const findBackgroundPosition = (scale, imageWidth, imageHeight, maskWidth, maskHeight, existingPokemon = []) => {
    const pokemonWidth = imageWidth * scale
    const pokemonHeight = imageHeight * scale
    
    // Scale mask coordinates to image coordinates
    const scaleX = imageWidth / maskWidth
    const scaleY = imageHeight / maskHeight
    
    // Check if Pokemon fits in image
    const maxX = imageWidth - pokemonWidth
    const maxY = imageHeight - pokemonHeight
    
    if (maxX <= 0 || maxY <= 0) {
      // Pokemon too large
      return null
    }
    
    const categoryMasksReady =
      skyMask &&
      waterMask &&
      grassMask &&
      groundMask &&
      otherMask &&
      skyMask.width === maskWidth &&
      skyMask.height === maskHeight

    // Helper function to check if a point is in background
    const isBackgroundPoint = (x, y) => {
      const maskX = Math.floor(x / scaleX)
      const maskY = Math.floor(y / scaleY)
      
      if (maskX >= 0 && maskX < maskWidth && maskY >= 0 && maskY < maskHeight) {
        const maskIndex = maskY * maskWidth + maskX
        return backgroundMask.data[maskIndex] === 1
      }
      return false
    }

    /** Dominant background category (sky/water/grass/ground/other) under the placement box */
    const getDominantBgCategory = (x, y) => {
      if (!categoryMasksReady) return 'other'
      const sampleDensity = Math.max(5, Math.min(pokemonWidth, pokemonHeight) / 10)
      const stepX = Math.max(1, pokemonWidth / sampleDensity)
      const stepY = Math.max(1, pokemonHeight / sampleDensity)
      const counts = { sky: 0, water: 0, grass: 0, ground: 0, other: 0 }
      for (let checkY = y; checkY < y + pokemonHeight; checkY += stepY) {
        for (let checkX = x; checkX < x + pokemonWidth; checkX += stepX) {
          const maskX = Math.floor(checkX / scaleX)
          const maskY = Math.floor(checkY / scaleY)
          if (maskX < 0 || maskX >= maskWidth || maskY < 0 || maskY >= maskHeight) continue
          const maskIndex = maskY * maskWidth + maskX
          if (backgroundMask.data[maskIndex] !== 1) continue
          if (skyMask.data[maskIndex]) counts.sky++
          else if (waterMask.data[maskIndex]) counts.water++
          else if (grassMask.data[maskIndex]) counts.grass++
          else if (groundMask.data[maskIndex]) counts.ground++
          else if (otherMask.data[maskIndex]) counts.other++
        }
      }
      let best = 'other'
      let bestN = -1
      for (const [k, v] of Object.entries(counts)) {
        if (v > bestN) {
          bestN = v
          best = k
        }
      }
      return bestN <= 0 ? 'other' : best
    }
    
    // Helper function to check if entire Pokemon area is in background
    // Checks a small area around the position for any non-background pixels
    const isPositionValid = (x, y) => {
      // Define the area to check - the Pokemon's bounding box
      const checkAreaWidth = pokemonWidth
      const checkAreaHeight = pokemonHeight
      
      // Sample points in a grid pattern across the area
      // Use a reasonable sampling density (check every N pixels)
      const sampleDensity = Math.max(5, Math.min(pokemonWidth, pokemonHeight) / 10) // Sample every 5-10% of size
      const stepX = Math.max(1, checkAreaWidth / sampleDensity)
      const stepY = Math.max(1, checkAreaHeight / sampleDensity)
      
      // Check all points in the area
      for (let checkY = y; checkY < y + checkAreaHeight; checkY += stepY) {
        for (let checkX = x; checkX < x + checkAreaWidth; checkX += stepX) {
          // If any point is NOT in background, reject this position
          if (!isBackgroundPoint(checkX, checkY)) {
            return false
          }
        }
      }
      
      // All checked points are in background - position is valid
      return true
    }
    
    // Try to find a valid background position
    const maxAttempts = 500 // Increased attempts for better coverage
    const backgroundPositions = []
    
    // First, collect all potential background positions
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const randomX = Math.random() * maxX
      const randomY = Math.random() * maxY
      
      if (isPositionValid(randomX, randomY)) {
        // Check collision with existing Pokemon
        if (!checkCollisionWithExisting(randomX, randomY, pokemonWidth, pokemonHeight, existingPokemon)) {
          backgroundPositions.push({ x: randomX, y: randomY })
        }
      }
    }
    
    // If we found valid positions, return a random one with dominant bg category for type matching
    if (backgroundPositions.length > 0) {
      const randomIndex = Math.floor(Math.random() * backgroundPositions.length)
      const { x, y } = backgroundPositions[randomIndex]
      return { x, y, bgCategory: getDominantBgCategory(x, y) }
    }
    
    // If no valid position found, return null (don't place Pokemon)
    return null
  }

  const handleGenerate = async () => {
    if (!baseImage || !backgroundMask || !imageRef.current) return
    
    setIsLoading(true)
    try {
      // Generate random number of Pokemon (1-5)
      const numPokemon =
        pokemonAmount === 3 ? Math.floor(Math.random() * 6) + 1 : pokemonAmount
      
      // Get image dimensions
      const imageWidth = imageRef.current.naturalWidth || imageRef.current.width
      const imageHeight = imageRef.current.naturalHeight || imageRef.current.height
      
      const newPokemonList = []
      const existingPokemonForCollision = [] // Track placed Pokemon for collision detection
      
      // Generate and place each Pokemon (type must match dominant background at placement)
      const maxPlacementAttempts = 30
      const maxSpeciesAttemptsPerPlacement = 60

      for (let i = 0; i < numPokemon; i++) {
        let placed = false

        for (let placementAttempt = 0; placementAttempt < maxPlacementAttempts && !placed; placementAttempt++) {
          const randomScale = Math.random() * 0.4 + 0.2

          let position = findBackgroundPosition(
            randomScale,
            imageWidth,
            imageHeight,
            backgroundMask.width,
            backgroundMask.height,
            existingPokemonForCollision
          )

          let currentScale = randomScale
          if (!position) {
            for (let scaleAttempt = 0; scaleAttempt < 5; scaleAttempt++) {
              currentScale = currentScale * 0.8
              if (currentScale < 0.05) break

              position = findBackgroundPosition(
                currentScale,
                imageWidth,
                imageHeight,
                backgroundMask.width,
                backgroundMask.height,
                existingPokemonForCollision
              )

              if (position) break
            }
          }

          if (!position) continue

          const bgCategory = position.bgCategory

          for (let s = 0; s < maxSpeciesAttemptsPerPlacement && !placed; s++) {
            const randomPokemonId = Math.floor(Math.random() * 1024) + 1
            try {
              const response = await fetch(`https://pokeapi.co/api/v2/pokemon/${randomPokemonId}/`)
              if (!response.ok) continue

              const data = await response.json()

              if (!pokemonMatchesBackgroundCategory(data.types, bgCategory)) continue

              const imageUrl =
                data.sprites.other?.['official-artwork']?.front_default ||
                data.sprites.front_default

              if (!imageUrl) continue

              const pokemonWidth = imageWidth * currentScale
              const pokemonHeight = imageHeight * currentScale

              const leftPercent = (position.x / imageWidth) * 100
              const topPercent = (position.y / imageHeight) * 100

              newPokemonList.push({
                imageUrl,
                scale: currentScale,
                left: `${leftPercent}%`,
                top: `${topPercent}%`
              })

              existingPokemonForCollision.push({
                x: position.x,
                y: position.y,
                width: pokemonWidth,
                height: pokemonHeight
              })

              placed = true
            } catch (error) {
              console.error(`Error fetching Pokemon (slot ${i + 1}):`, error)
            }
          }
        }
      }
      
      if (newPokemonList.length > 0) {
        setPokemonList(newPokemonList)
      } else {
        alert(
          'Could not find suitable background areas for any Pokemon. Try another image or generate again.'
        )
      }
    } catch (error) {
      console.error('Error generating Pokemon:', error)
      alert('Failed to generate Pokemon. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <div className="app-container">
        <h1>Pokéfy Your Image</h1>
        {!baseImage ? (
          <button onClick={handleButtonClick} className="select-button">
            Select Image
          </button>
        ) : (
          <>
            {isProcessingBackground && (
              <div className="processing-message">
                Processing background (subject, sky, water, grass, ground, other)... This may take a moment.
              </div>
            )}
            <div className="image-view">
              <div
                className="image-stage"
                onMouseEnter={() => setShowMaskOverlay(true)}
                onMouseLeave={() => setShowMaskOverlay(false)}
                onTouchStart={() => setShowMaskOverlay(true)}
                onTouchEnd={() => setShowMaskOverlay(false)}
                onTouchCancel={() => setShowMaskOverlay(false)}
              >
                <img
                  ref={imageRef}
                  src={baseImage}
                  alt="Selected"
                  className="selected-image"
                  onLoad={updateImageLayout}
                />
                {imageLayout && (
                  <div
                    className="image-content-overlay"
                    style={{
                      left: imageLayout.offX,
                      top: imageLayout.offY,
                      width: imageLayout.drawW,
                      height: imageLayout.drawH,
                    }}
                  >
                    {showMaskOverlay && backgroundMask && (
                      <canvas ref={maskCanvasRef} className="mask-overlay" />
                    )}
                    {pokemonList.map((pokemon, index) => (
                      <div
                        key={`${pokemon.imageUrl}-${index}`}
                        className="pokemon-slot"
                        style={{
                          left: pokemon.left,
                          top: pokemon.top,
                          width: `${pokemon.scale * 100}%`,
                          height: `${pokemon.scale * 100}%`,
                        }}
                      >
                        <img
                          src={pokemon.imageUrl}
                          alt=""
                          className="pokemon-overlay-img"
                          draggable={false}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {backgroundMask && (
                <>
                  <div
                    className="mask-legend"
                    role="group"
                    aria-label="Segmentation overlay colors"
                  >
                    {Object.values(MASK_OVERLAY).map((entry) => (
                      <span key={entry.label} className="mask-legend-item">
                        <span
                          className="mask-legend-swatch"
                          style={{ backgroundColor: entry.css }}
                          aria-hidden
                        />
                        <span className="mask-legend-text">{entry.label}</span>
                      </span>
                    ))}
                  </div>
                  <p className="mask-preview-hint">
                    Hover the photo (desktop) or touch and hold (mobile) to preview the
                    segmentation overlay.
                  </p>
                </>
              )}
            </div>
            <div className="controls">
              <div className="slider-container">
                <label htmlFor="pokemon-amount-slider">
                  Pokémon:{' '}
                  {pokemonAmount === 3 ? 'Random (1–6)' : pokemonAmount}
                </label>
                <input
                  id="pokemon-amount-slider"
                  type="range"
                  min="1"
                  max="6"
                  step="1"
                  value={pokemonAmount}
                  onChange={(e) => setPokemonAmount(parseInt(e.target.value, 10))}
                  className="threshold-slider"
                  disabled={isProcessingBackground}
                />
                <div className="slider-hint">
                  Center = random count; 1–2 and 4–6 pick that many Pokémon
                </div>
              </div>
            </div>
            <div className="button-group">
              <button onClick={handleBack} className="back-button">
                Back
              </button>
              <button 
                onClick={handleGenerate} 
                className="generate-button"
                disabled={isLoading || isProcessingBackground || !backgroundMask}
              >
                {isLoading ? 'Loading...' : 'Generate'}
              </button>
            </div>
          </>
        )}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageSelect}
          accept="image/*"
          style={{ display: 'none' }}
        />
      </div>
    </>
  )
}

export default App
