import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, ChevronDown, ChevronRight, PencilIcon, Play, Search, Trash2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { api, type GeneratedImage, type GeneratedOutput, type ModelFamily } from '../api'
import { modelColor } from '@/lib/utils'
import { useLocalStorage } from '../hooks/useLocalStorage'
import type { GenerateFormState } from './generate'
import { findModelFamily, modelDefaults, randomSeed } from './generate/generate-state'
import { ImageDetail } from './ImageDetail'
import { LazyImage } from './LazyImage'

type Props = {
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  setActiveTab: (tab: 'train' | 'generate' | 'outputs' | 'datasets') => void
}

/** Shorten a model id for display: "flux-schnell" → "flux-schnell", keep it short */
function modelLabel(id: string): string {
  return id.length > 16 ? id.slice(0, 14) + '...' : id
}

export function OutputsGallery({ setForm, setActiveTab }: Props) {
  const queryClient = useQueryClient()

  const {
    data: groups = [],
    error,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['outputs'],
    queryFn: api.outputs,
    staleTime: 30_000,
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteOutput,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['outputs'] })
    },
  })

  const batchDeleteMutation = useMutation({
    mutationFn: api.batchDeleteOutputs,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['outputs'] })
      setSelectedPaths(new Set())
      setSelectMode(false)
    },
  })

  const favoriteMutation = useMutation<{ favorited: boolean }, Error, string, { prev: GeneratedOutput[] | undefined }>({
    mutationFn: (path: string) => api.favoriteOutput(path),
    onMutate: async (path: string) => {
      await queryClient.cancelQueries({ queryKey: ['outputs'] })
      const prev = queryClient.getQueryData<GeneratedOutput[]>(['outputs'])
      queryClient.setQueryData<GeneratedOutput[]>(['outputs'], (old) =>
        old?.map((g) => ({
          ...g,
          images: g.images.map((img) =>
            img.path === path ? { ...img, favorited: !img.favorited } : img,
          ),
        }))
      )
      return { prev }
    },
    onError: (_err, _path, context) => {
      if (context?.prev) {
        queryClient.setQueryData(['outputs'], context.prev)
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['outputs'] })
    },
  })

  // Models + families (cached from other tabs, needed for send-to-edit)
  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60_000,
  })
  const models = modelsResponse?.models ?? []

  const { data: families = [] } = useQuery<ModelFamily[]>({
    queryKey: ['model-families'],
    queryFn: api.modelFamilies,
    staleTime: 60 * 60_000,
  })

  const sendToEdit = useCallback((imageUrl: string, serverPath: string) => {
    setForm((prev) => {
      let modelId = prev.base_model_id
      let steps = prev.steps
      let guidance = prev.guidance
      const editModel = models.find(
        (m) => m.id === 'qwen-image-edit' || m.name.toLowerCase().includes('qwen-image-edit'),
      )
      if (editModel) {
        modelId = editModel.id
        const info = findModelFamily(editModel.name, families)
        const defaults = modelDefaults(editModel.name, info)
        steps = defaults.steps
        guidance = defaults.guidance
      }
      return {
        ...prev,
        mode: 'edit' as const,
        base_model_id: modelId,
        steps,
        guidance,
        edit_images: [{ type: 'server' as const, preview: imageUrl, serverPath }],
      }
    })
    setActiveTab('generate')
  }, [models, families, setForm, setActiveTab])

  const [selected, setSelected] = useState<GeneratedImage | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<GeneratedImage | null>(null)
  const [modelFilter, setModelFilter] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<string | null>(null)
  const [sortNewestFirst, setSortNewestFirst] = useLocalStorage('modl:outputs-sort', true)
  const [gridSize, setGridSize] = useLocalStorage<'s' | 'm' | 'l'>('modl:outputs-grid', 'm')
  const [favoriteFilter, setFavoriteFilter] = useState(false)
  const [groupBy, setGroupBy] = useLocalStorage<'date' | 'model' | 'none'>('modl:outputs-group', 'date')
  const [searchQuery, setSearchQuery] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  // Batch selection
  const [selectMode, setSelectMode] = useState(false)
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const [confirmBatchDelete, setConfirmBatchDelete] = useState(false)

  // Track cursor index for shift+click and shift+arrow range select
  const lastClickedRef = useRef<number>(-1)
  const [cursorIndex, setCursorIndex] = useState(-1)

  // Collapsed groups
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const toggleCollapsed = useCallback((key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])

  const gridClass = {
    s: 'grid-cols-3 gap-1.5 md:grid-cols-5 lg:grid-cols-7 xl:grid-cols-9',
    m: 'grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6',
    l: 'grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  }[gridSize]

  // Thumbnail width based on grid size — smaller grid = smaller thumbs
  const thumbWidth = { s: 200, m: 320, l: 480 }[gridSize]

  const modelOptions = useMemo(() => {
    const models = new Set<string>()
    for (const group of groups) {
      for (const image of group.images) {
        if (image.base_model_id) {
          models.add(image.base_model_id)
        }
      }
    }
    return [...models].sort((a, b) => a.localeCompare(b))
  }, [groups])

  const dateOptions = useMemo(() => {
    return groups.map((group) => group.date)
  }, [groups])

  const searchLower = searchQuery.toLowerCase().trim()

  const filteredGroups = useMemo(() => {
    // 1. Flatten all images, apply filters + search
    let allImages: (GeneratedImage & { date: string })[] = []
    for (const group of groups) {
      if (dateFilter && group.date !== dateFilter) continue
      for (const image of group.images) {
        if (modelFilter && image.base_model_id !== modelFilter) continue
        if (favoriteFilter && !image.favorited) continue
        if (searchLower && !(
          image.prompt?.toLowerCase().includes(searchLower) ||
          image.filename.toLowerCase().includes(searchLower) ||
          image.base_model_id?.toLowerCase().includes(searchLower) ||
          image.lora_name?.toLowerCase().includes(searchLower)
        )) continue
        allImages.push({ ...image, date: group.date })
      }
    }

    // 2. Sort
    allImages.sort((a, b) => {
      const left = a.modified ?? 0
      const right = b.modified ?? 0
      return sortNewestFirst ? right - left : left - right
    })

    // 3. Group
    const grouped = new Map<string, GeneratedImage[]>()
    for (const img of allImages) {
      const key = groupBy === 'model'
        ? (img.base_model_id ?? 'Unknown model')
        : groupBy === 'date'
          ? img.date
          : 'All images'
      if (!grouped.has(key)) grouped.set(key, [])
      grouped.get(key)!.push(img)
    }

    // 4. Sort groups
    const result: GeneratedOutput[] = []
    for (const [key, images] of grouped) {
      result.push({ date: key, images })
    }
    if (groupBy !== 'none') {
      result.sort((a, b) => sortNewestFirst ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date))
    }
    return result
  }, [groups, dateFilter, modelFilter, sortNewestFirst, favoriteFilter, groupBy, searchLower])

  const allFilteredImages = useMemo(() => {
    return filteredGroups.flatMap((g) => g.images)
  }, [filteredGroups])

  const totalImageCount = useMemo(() => {
    return groups.reduce((sum, g) => sum + g.images.length, 0)
  }, [groups])

  const isFiltered = modelFilter !== null || dateFilter !== null || favoriteFilter || searchLower !== ''

  // ── Stats bar data (computed from visible/filtered images) ──────────
  const [statsOpen, setStatsOpen] = useLocalStorage('modl:outputs-stats', true)

  const stats = useMemo(() => {
    const images = allFilteredImages
    const total = images.length
    const today = new Date().toISOString().slice(0, 10) // YYYY-MM-DD

    // By model: count + star count
    const byModel = new Map<string, { count: number; stars: number }>()
    let todayCount = 0

    for (const img of images) {
      const mid = img.base_model_id ?? 'unknown'
      const entry = byModel.get(mid) ?? { count: 0, stars: 0 }
      entry.count++
      if (img.favorited) entry.stars++
      byModel.set(mid, entry)

      // Check if image is from today (use modified timestamp or path date)
      if (img.path?.includes(today)) todayCount++
    }

    // Sort by count descending
    const modelChips = [...byModel.entries()]
      .sort((a, b) => b[1].count - a[1].count)

    return { total, todayCount, modelChips }
  }, [allFilteredImages])

  // Global keyboard shortcuts: "/" to search, arrow keys for select mode navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        searchRef.current?.focus()
        return
      }
      // Arrow key navigation in select mode
      if (!selectMode || selected) return
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      e.preventDefault()
      const total = allFilteredImages.length
      if (total === 0) return
      setCursorIndex((prev) => {
        const cur = prev < 0 ? 0 : prev
        let next = cur
        if (e.key === 'ArrowRight') next = Math.min(cur + 1, total - 1)
        else if (e.key === 'ArrowLeft') next = Math.max(cur - 1, 0)
        else if (e.key === 'ArrowDown') next = Math.min(cur + 4, total - 1)
        else if (e.key === 'ArrowUp') next = Math.max(cur - 4, 0)
        if (e.shiftKey) {
          const anchor = lastClickedRef.current >= 0 ? lastClickedRef.current : 0
          const start = Math.min(anchor, next)
          const end = Math.max(anchor, next)
          setSelectedPaths((prevSet) => {
            const s = new Set(prevSet)
            for (let i = start; i <= end; i++) s.add(allFilteredImages[i].path)
            return s
          })
        }
        return next
      })
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectMode, selected, allFilteredImages])

  const togglePathSelection = useCallback((path: string, shiftKey = false) => {
    const clickedIndex = allFilteredImages.findIndex((img) => img.path === path)
    if (shiftKey && lastClickedRef.current >= 0 && clickedIndex >= 0) {
      // Range select between last clicked and current
      const start = Math.min(lastClickedRef.current, clickedIndex)
      const end = Math.max(lastClickedRef.current, clickedIndex)
      setSelectedPaths((prev) => {
        const next = new Set(prev)
        for (let i = start; i <= end; i++) {
          next.add(allFilteredImages[i].path)
        }
        return next
      })
    } else {
      setSelectedPaths((prev) => {
        const next = new Set(prev)
        if (next.has(path)) {
          next.delete(path)
        } else {
          next.add(path)
        }
        return next
      })
    }
    lastClickedRef.current = clickedIndex
    setCursorIndex(clickedIndex)
  }, [allFilteredImages])

  const selectAllVisible = useCallback(() => {
    setSelectedPaths(new Set(allFilteredImages.map((img) => img.path)))
  }, [allFilteredImages])

  const deselectAll = useCallback(() => {
    setSelectedPaths(new Set())
  }, [])

  const onDelete = async (image: GeneratedImage) => {
    await deleteMutation.mutateAsync({ artifact_id: image.artifact_id, path: image.path })
    setDeleteTarget(null)
  }

  const onBatchDelete = async () => {
    const items = allFilteredImages
      .filter((img) => selectedPaths.has(img.path))
      .map((img) => ({ artifact_id: img.artifact_id, path: img.path }))
    if (items.length === 0) return
    await batchDeleteMutation.mutateAsync(items)
    setConfirmBatchDelete(false)
  }

  const combinedError = error ?? deleteMutation.error ?? favoriteMutation.error ?? batchDeleteMutation.error

  return (
    <div className="space-y-4">
      {/* Sticky toolbar area */}
      <div className="sticky top-14 z-20 -mx-4 -mt-6 px-4 pt-4 pb-2 bg-[#09090e]/95 backdrop-blur md:top-0 md:-mx-6 md:px-6 md:-mt-6 md:pt-6">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Search */}
          <div className="relative">
            <Search className="pointer-events-none absolute left-2 top-1/2 size-3 -translate-y-1/2 text-muted-foreground" />
            <input
              ref={searchRef}
              type="text"
              placeholder="Search prompts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 w-40 rounded-md border border-border/50 bg-transparent pl-7 pr-6 text-xs text-foreground placeholder:text-muted-foreground/60 focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring/50"
            />
            {searchQuery && (
              <button
                type="button"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                onClick={() => { setSearchQuery(''); searchRef.current?.focus() }}
              >
                <X className="size-3" />
              </button>
            )}
          </div>

          <Select
            value={modelFilter ?? '__all__'}
            onValueChange={(v) => setModelFilter(v === '__all__' ? null : v)}
          >
            <SelectTrigger size="sm" className="h-7 min-w-[100px] gap-1.5 border-border/50 bg-transparent px-2 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All models</SelectItem>
              {modelOptions.map((model) => (
                <SelectItem key={model} value={model}>
                  <span
                    className="mr-1.5 inline-block size-2 rounded-full"
                    style={{ backgroundColor: modelColor(model) }}
                  />
                  {model}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={dateFilter ?? '__all__'}
            onValueChange={(v) => setDateFilter(v === '__all__' ? null : v)}
          >
            <SelectTrigger size="sm" className="h-7 min-w-[100px] gap-1.5 border-border/50 bg-transparent px-2 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All dates</SelectItem>
              {dateOptions.map((date) => (
                <SelectItem key={date} value={date}>{date}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            type="button"
            size="sm"
            variant={favoriteFilter ? 'secondary' : 'ghost'}
            className={`h-7 px-2.5 text-xs ${favoriteFilter ? 'text-yellow-400' : ''}`}
            onClick={() => setFavoriteFilter((prev) => !prev)}
          >
            Starred
          </Button>

          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={() => setSortNewestFirst((prev) => !prev)}>
            {sortNewestFirst ? '↓ Newest' : '↑ Oldest'}
          </Button>

        </div>

        <div className="flex items-center gap-1.5">
          {/* Group by */}
          <Select value={groupBy} onValueChange={(v) => setGroupBy(v as 'date' | 'model' | 'none')}>
            <SelectTrigger size="sm" className="h-7 gap-1.5 border-border/50 bg-transparent px-2 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="date">Group by date</SelectItem>
              <SelectItem value="model">Group by model</SelectItem>
              <SelectItem value="none">No grouping</SelectItem>
            </SelectContent>
          </Select>

          {/* Select mode toggle */}
          <Button
            type="button"
            size="sm"
            variant={selectMode ? 'secondary' : 'ghost'}
            className="h-7 px-2.5 text-xs"
            onClick={() => {
              setSelectMode((prev) => !prev)
              if (selectMode) {
                setSelectedPaths(new Set())
                setConfirmBatchDelete(false)
                setCursorIndex(-1)
                lastClickedRef.current = -1
              }
            }}
          >
            {selectMode ? <><Check className="mr-1 size-3" />Select</> : 'Select'}
          </Button>

          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={() => void refetch()}>
            {isFetching ? 'Refreshing...' : 'Refresh'}
          </Button>
          <div className="flex items-center rounded-md border border-border/50">
            {(['s', 'm', 'l'] as const).map((size) => (
              <button
                key={size}
                type="button"
                onClick={() => setGridSize(size)}
                className={`h-7 w-7 text-[10px] font-semibold uppercase transition-colors first:rounded-l-md last:rounded-r-md ${
                  gridSize === size
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {size}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Batch selection toolbar */}
      {selectMode && (
        <div className="flex items-center gap-2 rounded-lg border border-border/50 bg-secondary/30 px-3 py-2">
          <span className="text-xs text-muted-foreground">
            {selectedPaths.size} selected
          </span>
          <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={selectAllVisible}>
            Select all ({allFilteredImages.length})
          </Button>
          {selectedPaths.size > 0 && (
            <>
              <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={deselectAll}>
                Deselect
              </Button>
              {!confirmBatchDelete ? (
                <Button
                  type="button"
                  size="sm"
                  variant="destructive"
                  className="h-6 px-2 text-xs"
                  onClick={() => setConfirmBatchDelete(true)}
                  disabled={batchDeleteMutation.isPending}
                >
                  <Trash2 className="mr-1 size-3" />
                  Delete {selectedPaths.size}
                </Button>
              ) : (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-destructive">
                    Delete {selectedPaths.size} images?
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    className="h-6 px-2 text-xs"
                    onClick={() => void onBatchDelete()}
                    disabled={batchDeleteMutation.isPending}
                  >
                    {batchDeleteMutation.isPending ? 'Deleting...' : 'Confirm'}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={() => setConfirmBatchDelete(false)}
                  >
                    <X className="size-3" />
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      )}
      {/* Stats bar */}
      {!isLoading && stats.total > 0 && (
        <div className="mt-2">
          <button
            type="button"
            className="flex items-center gap-1 text-[10px] text-muted-foreground/50 hover:text-muted-foreground/80 transition-colors"
            onClick={() => setStatsOpen((prev) => !prev)}
          >
            {statsOpen
              ? <ChevronDown className="size-2.5" />
              : <ChevronRight className="size-2.5" />
            }
            Stats
          </button>
          {statsOpen && (
            <div className="mt-1 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-xs text-muted-foreground/70">
              <span>
                <span className="font-medium text-foreground/80">{stats.total}</span> image{stats.total !== 1 ? 's' : ''}
                {isFiltered && <span className="text-muted-foreground/40"> / {totalImageCount}</span>}
              </span>

              {stats.todayCount > 0 && (
                <span>
                  <span className="font-medium text-foreground/80">{stats.todayCount}</span> today
                </span>
              )}

              <span className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
                {stats.modelChips.map(([mid, { count, stars }], i) => (
                  <span key={mid} className="inline-flex items-center gap-1">
                    <span
                      className="inline-block size-1.5 rounded-full"
                      style={{ backgroundColor: modelColor(mid) }}
                    />
                    <span className="text-foreground/70">{modelLabel(mid)}</span>
                    <span className="tabular-nums">{count}</span>
                    {stars > 0 && <span className="text-yellow-500/80">★{stars}</span>}
                    {i < stats.modelChips.length - 1 && <span className="text-muted-foreground/20">·</span>}
                  </span>
                ))}
              </span>
            </div>
          )}
        </div>
      )}
      </div>{/* end sticky toolbar area */}

      {combinedError ? (
        <p className="text-sm text-destructive">Failed to load outputs: {String(combinedError)}</p>
      ) : null}

      {isLoading ? (
        <div className={`grid ${gridClass}`}>
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="aspect-square animate-pulse rounded-lg bg-secondary/50" />
          ))}
        </div>
      ) : null}

      {!isLoading && filteredGroups.length === 0 && !combinedError ? (
        <p className="py-8 text-center text-sm text-muted-foreground">No generated images for the selected filters.</p>
      ) : null}

      {filteredGroups.map((group) => {
        const isCollapsed = collapsedGroups.has(group.date)
        return (
        <section key={group.date} className="space-y-2" style={!isCollapsed ? { contentVisibility: 'auto', containIntrinsicSize: 'auto 400px' } : undefined}>
          {groupBy !== 'none' && (
            <button
              type="button"
              className="flex w-full items-center gap-1.5 text-left"
              onClick={() => toggleCollapsed(group.date)}
            >
              {isCollapsed
                ? <ChevronRight className="size-3.5 text-muted-foreground/60" />
                : <ChevronDown className="size-3.5 text-muted-foreground/60" />
              }
              {groupBy === 'model' && (
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ backgroundColor: modelColor(group.date) }}
                />
              )}
              <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{group.date}</span>
              <span className="text-[10px] text-muted-foreground/50">{group.images.length}</span>
            </button>
          )}
          {!isCollapsed && <div className={`grid ${gridClass}`}>
            {group.images.map((image) => {
              const isSelected = selectedPaths.has(image.path)
              const isDeleting = deleteTarget?.path === image.path
              const isCursor = selectMode && cursorIndex >= 0 && allFilteredImages[cursorIndex]?.path === image.path

              return (
                <article key={image.path} className={`group relative overflow-hidden rounded-lg ${isSelected ? 'ring-2 ring-primary' : ''} ${isCursor && !isSelected ? 'ring-2 ring-primary/40' : ''}`}>
                  <LazyImage
                    src={`/files/${image.path}`}
                    alt={image.filename}
                    className="aspect-square"
                    thumbWidth={thumbWidth}
                    onClick={(e) => {
                      if (selectMode) {
                        togglePathSelection(image.path, e.shiftKey)
                      } else {
                        setSelected(image)
                      }
                    }}
                  />
                  {/* Hover overlay */}
                  <div className="pointer-events-none absolute inset-0 bg-black/0 transition-colors group-hover:bg-black/30" />

                  {/* Model color dot */}
                  {image.base_model_id && (
                    <div
                      className="absolute bottom-1.5 left-1.5 flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5 backdrop-blur-sm"
                      title={image.base_model_id}
                    >
                      <span
                        className="inline-block size-2 rounded-full"
                        style={{ backgroundColor: modelColor(image.base_model_id) }}
                      />
                      <span className="text-[10px] font-medium leading-none text-white/90">
                        {modelLabel(image.base_model_id)}
                      </span>
                    </div>
                  )}

                  {/* Select checkbox (in select mode) */}
                  {selectMode && (
                    <button
                      type="button"
                      className="absolute top-1.5 left-1.5 drop-shadow"
                      onClick={(e) => {
                        e.stopPropagation()
                        togglePathSelection(image.path, e.shiftKey)
                      }}
                    >
                      <div className={`flex size-5 items-center justify-center rounded border-2 transition-colors ${
                        isSelected
                          ? 'border-primary bg-primary'
                          : 'border-white/80 bg-black/40'
                      }`}>
                        {isSelected && <Check className="size-3.5 text-white" strokeWidth={3} />}
                      </div>
                    </button>
                  )}

                  {/* Favorite button (when not in select mode) */}
                  {!selectMode && (
                    <button
                      type="button"
                      className={`absolute top-1.5 left-1.5 text-base leading-none drop-shadow transition-opacity ${
                        image.favorited
                          ? 'opacity-100'
                          : 'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation()
                        favoriteMutation.mutate(image.path)
                      }}
                    >
                      {image.favorited ? '⭐' : '☆'}
                    </button>
                  )}

                  {/* Inline delete confirmation */}
                  {isDeleting && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/70 backdrop-blur-sm">
                      <p className="text-xs font-medium text-white">Delete?</p>
                      <div className="flex gap-1.5">
                        <Button
                          type="button"
                          size="sm"
                          variant="destructive"
                          className="h-7 px-3 text-xs"
                          disabled={deleteMutation.isPending}
                          onClick={() => void onDelete(image)}
                        >
                          {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="secondary"
                          className="h-7 px-3 text-xs"
                          onClick={() => setDeleteTarget(null)}
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Hover action buttons (not in select mode) */}
                  {!selectMode && !isDeleting && (
                    <div className="pointer-events-none absolute top-1.5 right-1.5 flex gap-1 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100">
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        className="h-6 w-6 p-0"
                        title="Open as recipe"
                        onClick={(e) => {
                          e.stopPropagation()
                          setForm((prev) => ({
                            ...prev,
                            prompt: image.prompt ?? '',
                            base_model_id: image.base_model_id ?? prev.base_model_id,
                            loras: image.lora_name ? [{ id: image.lora_name, name: image.lora_name, strength: image.lora_strength ?? 1.0, enabled: true }] : [],
                            seed: image.seed ?? randomSeed(),
                            steps: image.steps ?? 20,
                            guidance: image.guidance ?? 3.5,
                            width: image.width ?? 1024,
                            height: image.height ?? 1024,
                          }))
                          setActiveTab('generate')
                        }}
                      >
                        <Play className="size-3" />
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        className="h-6 w-6 p-0"
                        title="Send to Edit"
                        onClick={(e) => {
                          e.stopPropagation()
                          sendToEdit(`/files/${image.path}`, image.path)
                        }}
                      >
                        <PencilIcon className="size-3" />
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        className="h-6 px-2 text-[10px]"
                        onClick={() => setSelected(image)}
                      >
                        Info
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="destructive"
                        className="h-6 w-6 p-0"
                        onClick={(e) => { e.stopPropagation(); setDeleteTarget(image) }}
                        disabled={deleteMutation.isPending}
                      >
                        <Trash2 className="size-3" />
                      </Button>
                    </div>
                  )}
                </article>
              )
            })}
          </div>}
        </section>
        )
      })}

      <ImageDetail
        image={selected}
        onClose={() => setSelected(null)}
        setForm={setForm}
        setActiveTab={setActiveTab}
        allImages={allFilteredImages}
        onNavigate={setSelected}
        onToggleFavorite={(path) => favoriteMutation.mutate(path)}
        onEditImage={sendToEdit}
        onDeleteImage={(img) => {
          setDeleteTarget(img)
          setSelected(null)
        }}
      />
    </div>
  )
}
