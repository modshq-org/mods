import { useEffect, useState } from 'react'

function resolveInitial<T>(initialValue: T | (() => T)): T {
  return initialValue instanceof Function ? initialValue() : initialValue
}

/**
 * Keys whose values cannot survive JSON round-tripping (File objects, Blobs, etc.).
 * They are stripped before writing and reset to defaults on read.
 */
const NON_SERIALIZABLE_KEYS = new Set(['init_image_file', 'edit_images', 'init_image'])

export function useLocalStorage<T>(key: string, initialValue: T | (() => T)) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    const fallback = resolveInitial(initialValue)

    if (typeof window === 'undefined') {
      return fallback
    }

    try {
      const item = window.localStorage.getItem(key)
      if (!item) return fallback
      const parsed = JSON.parse(item) as T
      // Merge with defaults so new fields are always present
      if (typeof fallback === 'object' && fallback !== null && !Array.isArray(fallback)) {
        const merged = { ...fallback, ...parsed }
        // Restore non-serializable fields to their defaults
        for (const k of NON_SERIALIZABLE_KEYS) {
          if (k in (fallback as Record<string, unknown>)) {
            ;(merged as Record<string, unknown>)[k] = (fallback as Record<string, unknown>)[k]
          }
        }
        return merged
      }
      return parsed
    } catch {
      return fallback
    }
  })

  useEffect(() => {
    try {
      let toStore: unknown = storedValue
      // Strip non-serializable fields before writing
      if (typeof storedValue === 'object' && storedValue !== null && !Array.isArray(storedValue)) {
        const copy = { ...storedValue } as Record<string, unknown>
        for (const k of NON_SERIALIZABLE_KEYS) {
          delete copy[k]
        }
        toStore = copy
      }
      window.localStorage.setItem(key, JSON.stringify(toStore))
    } catch {
      // Ignore write failures (quota/private mode).
    }
  }, [key, storedValue])

  return [storedValue, setStoredValue] as const
}
