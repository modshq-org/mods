// Standardized stale times for TanStack Query
// Grouped by data volatility

/** Real-time data — GPU status, queue info, active sidebar badges */
export const STALE_REALTIME = 4_000

/** Fast-changing data — outputs list, training runs list, active run details */
export const STALE_FAST = 30_000

/** Moderate data — models list, datasets, library loras, training wizard dropdowns */
export const STALE_MODERATE = 60_000

/** Slow-changing data — model families, installed model details */
export const STALE_SLOW = 5 * 60_000

/** Near-static data — architecture info, model parameter schemas */
export const STALE_STATIC = 60 * 60_000
