import { useMemo, useRef, useEffect } from 'react'
import type { LossPoint } from '../../api'

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export type LossChartProps = {
  points: LossPoint[]
}

// ---------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------

function emaSmooth(points: LossPoint[], alpha = 0.05): LossPoint[] {
  if (points.length === 0) return []
  const result: LossPoint[] = [points[0]]
  let ema = points[0].loss
  for (let i = 1; i < points.length; i++) {
    ema = alpha * points[i].loss + (1 - alpha) * ema
    result.push({ step: points[i].step, loss: ema })
  }
  return result
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LossChart({ points }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const smoothed = useMemo(() => emaSmooth(points), [points])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || smoothed.length < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height

    const pad = { top: 12, right: 12, bottom: 24, left: 48 }
    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom

    const minStep = smoothed[0].step
    const maxStep = smoothed[smoothed.length - 1].step
    const stepRange = maxStep - minStep || 1
    const smoothLosses = smoothed.map((p) => p.loss)
    const minLoss = Math.min(...smoothLosses) * 0.95
    const maxLoss = Math.max(...smoothLosses) * 1.05
    const lossRange = maxLoss - minLoss || 1

    const toX = (step: number) => pad.left + ((step - minStep) / stepRange) * plotW
    const toY = (loss: number) => pad.top + (1 - (loss - minLoss) / lossRange) * plotH

    ctx.clearRect(0, 0, w, h)

    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }

    ctx.strokeStyle = 'rgba(167, 139, 250, 0.15)'
    ctx.lineWidth = 1
    ctx.lineJoin = 'round'
    ctx.beginPath()
    for (let i = 0; i < points.length; i++) {
      const x = toX(points[i].step)
      const y = toY(Math.max(minLoss, Math.min(maxLoss, points[i].loss)))
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    ctx.strokeStyle = '#a78bfa'
    ctx.lineWidth = 2
    ctx.lineJoin = 'round'
    ctx.beginPath()
    for (let i = 0; i < smoothed.length; i++) {
      const x = toX(smoothed[i].step)
      const y = toY(smoothed[i].loss)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.font = '10px system-ui'
    ctx.textAlign = 'right'
    ctx.fillText(maxLoss.toFixed(3), pad.left - 4, pad.top + 10)
    ctx.fillText(minLoss.toFixed(3), pad.left - 4, h - pad.bottom)
    ctx.textAlign = 'center'
    ctx.fillText(`Step ${minStep}`, pad.left, h - 6)
    ctx.fillText(`Step ${maxStep}`, w - pad.right, h - 6)

    const lastSmooth = smoothed[smoothed.length - 1]
    ctx.fillStyle = '#a78bfa'
    ctx.textAlign = 'left'
    ctx.font = 'bold 11px system-ui'
    ctx.fillText(lastSmooth.loss.toFixed(4), toX(lastSmooth.step) + 4, toY(lastSmooth.loss) + 4)
  }, [points, smoothed])

  if (points.length < 2) {
    return <p className="text-xs text-muted-foreground">Not enough data for loss chart.</p>
  }

  return (
    <canvas
      ref={canvasRef}
      className="h-36 w-full"
      style={{ display: 'block' }}
    />
  )
}
