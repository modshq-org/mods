import type React from 'react'
import { Check, ChevronDown, ChevronRight, Search, Trash2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { modelColor } from '@/lib/utils'

/** Shorten a model id for display */
function modelLabel(id: string): string {
  return id.length > 16 ? id.slice(0, 14) + '...' : id
}

type StatsData = {
  total: number
  todayCount: number
  modelChips: [string, { count: number; stars: number }][]
}

export type GalleryFilterBarProps = {
  // Search
  searchQuery: string
  onSearchChange: (query: string) => void
  searchRef: React.RefObject<HTMLInputElement | null>

  // Model filter
  modelFilter: string | null
  onModelFilterChange: (model: string | null) => void
  modelOptions: string[]

  // Date filter
  dateFilter: string | null
  onDateFilterChange: (date: string | null) => void
  dateOptions: string[]

  // Starred
  favoriteFilter: boolean
  onFavoriteFilterChange: (starred: boolean) => void

  // Sort
  sortNewestFirst: boolean
  onSortChange: (newest: boolean) => void

  // Group by
  groupBy: 'date' | 'model' | 'none'
  onGroupByChange: (groupBy: 'date' | 'model' | 'none') => void

  // Select mode
  selectMode: boolean
  onSelectModeChange: (mode: boolean) => void

  // Grid size
  gridSize: 's' | 'm' | 'l'
  onGridSizeChange: (size: 's' | 'm' | 'l') => void

  // Refresh
  isFetching: boolean
  onRefresh: () => void

  // Batch selection
  selectedCount: number
  filteredCount: number
  onSelectAll: () => void
  onDeselectAll: () => void
  confirmBatchDelete: boolean
  onConfirmBatchDelete: (confirm: boolean) => void
  onBatchDelete: () => void
  batchDeletePending: boolean

  // Stats
  isLoading: boolean
  stats: StatsData
  statsOpen: boolean
  onStatsOpenChange: (open: boolean) => void
  isFiltered: boolean
  totalImageCount: number
}

export function GalleryFilterBar({
  searchQuery,
  onSearchChange,
  searchRef,
  modelFilter,
  onModelFilterChange,
  modelOptions,
  dateFilter,
  onDateFilterChange,
  dateOptions,
  favoriteFilter,
  onFavoriteFilterChange,
  sortNewestFirst,
  onSortChange,
  groupBy,
  onGroupByChange,
  selectMode,
  onSelectModeChange,
  gridSize,
  onGridSizeChange,
  isFetching,
  onRefresh,
  selectedCount,
  filteredCount,
  onSelectAll,
  onDeselectAll,
  confirmBatchDelete,
  onConfirmBatchDelete,
  onBatchDelete,
  batchDeletePending,
  isLoading,
  stats,
  statsOpen,
  onStatsOpenChange,
  isFiltered,
  totalImageCount,
}: GalleryFilterBarProps) {
  return (
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
              onChange={(e) => onSearchChange(e.target.value)}
              className="h-7 w-40 rounded-md border border-border/50 bg-transparent pl-7 pr-6 text-xs text-foreground placeholder:text-muted-foreground/60 focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring/50"
            />
            {searchQuery && (
              <button
                type="button"
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                onClick={() => {
                  onSearchChange('')
                  searchRef.current?.focus()
                }}
              >
                <X className="size-3" />
              </button>
            )}
          </div>

          <Select
            value={modelFilter ?? '__all__'}
            onValueChange={(v) => onModelFilterChange(v === '__all__' ? null : v)}
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
            onValueChange={(v) => onDateFilterChange(v === '__all__' ? null : v)}
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
            onClick={() => onFavoriteFilterChange(!favoriteFilter)}
          >
            Starred
          </Button>

          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={() => onSortChange(!sortNewestFirst)}>
            {sortNewestFirst ? '\u2193 Newest' : '\u2191 Oldest'}
          </Button>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Group by */}
          <Select value={groupBy} onValueChange={(v) => onGroupByChange(v as 'date' | 'model' | 'none')}>
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
            onClick={() => onSelectModeChange(!selectMode)}
          >
            {selectMode ? <><Check className="mr-1 size-3" />Select</> : 'Select'}
          </Button>

          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={onRefresh}>
            {isFetching ? 'Refreshing...' : 'Refresh'}
          </Button>
          <div className="flex items-center rounded-md border border-border/50">
            {(['s', 'm', 'l'] as const).map((size) => (
              <button
                key={size}
                type="button"
                onClick={() => onGridSizeChange(size)}
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
            {selectedCount} selected
          </span>
          <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onSelectAll}>
            Select all ({filteredCount})
          </Button>
          {selectedCount > 0 && (
            <>
              <Button type="button" size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onDeselectAll}>
                Deselect
              </Button>
              {!confirmBatchDelete ? (
                <Button
                  type="button"
                  size="sm"
                  variant="destructive"
                  className="h-6 px-2 text-xs"
                  onClick={() => onConfirmBatchDelete(true)}
                  disabled={batchDeletePending}
                >
                  <Trash2 className="mr-1 size-3" />
                  Delete {selectedCount}
                </Button>
              ) : (
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-destructive">
                    Delete {selectedCount} images?
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    className="h-6 px-2 text-xs"
                    onClick={onBatchDelete}
                    disabled={batchDeletePending}
                  >
                    {batchDeletePending ? 'Deleting...' : 'Confirm'}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={() => onConfirmBatchDelete(false)}
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
            onClick={() => onStatsOpenChange(!statsOpen)}
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
                    {stars > 0 && <span className="text-yellow-500/80">{'\u2605'}{stars}</span>}
                    {i < stats.modelChips.length - 1 && <span className="text-muted-foreground/20">{'\u00B7'}</span>}
                  </span>
                ))}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
