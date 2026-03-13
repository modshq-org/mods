import { useQuery } from '@tanstack/react-query'
import { api, type GpuStatus } from '../api'
import { STALE_REALTIME } from '@/lib/query-keys'

/**
 * Single source of truth for GPU status polling.
 * Deduplicated by TanStack Query's shared cache — safe to call from
 * multiple components without extra network requests.
 */
export function useGpuStatus() {
  return useQuery<GpuStatus>({
    queryKey: ['gpu'],
    queryFn: api.gpu,
    refetchInterval: 5000,
    staleTime: STALE_REALTIME,
  })
}
