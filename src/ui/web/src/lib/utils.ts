import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// ── Stable model colors (hash-based, not index-based) ────────────────
const MODEL_COLORS = [
  '#3b82f6', // blue
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#f97316', // orange
  '#10b981', // emerald
  '#06b6d4', // cyan
  '#eab308', // yellow
  '#ef4444', // red
  '#6366f1', // indigo
  '#14b8a6', // teal
]

/** Deterministic color for a model id — stable across sessions and filter changes. */
export function modelColor(modelId: string | undefined): string {
  if (!modelId) return '#6b7280'
  // Simple string hash → palette index
  let hash = 0
  for (let i = 0; i < modelId.length; i++) {
    hash = ((hash << 5) - hash + modelId.charCodeAt(i)) | 0
  }
  return MODEL_COLORS[Math.abs(hash) % MODEL_COLORS.length]
}
