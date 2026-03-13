import { lazy, Suspense } from 'react'
import { useLocation, useSearch } from 'wouter'
import { AppSidebar } from './components/AppSidebar'
import { GenerateView } from './components/generate'
import { ErrorBoundary } from './components/ErrorBoundary'
import { MobileNav } from './components/MobileNav'
import { QueuePanel } from './components/QueuePanel'
import { useLocalStorage } from './hooks/useLocalStorage'
import { FormProvider } from './contexts/FormContext'

const TrainingRuns = lazy(() => import('./components/TrainingRuns').then(m => ({ default: m.TrainingRuns })))
const OutputsGallery = lazy(() => import('./components/OutputsGallery').then(m => ({ default: m.OutputsGallery })))
const DatasetViewer = lazy(() => import('./components/DatasetViewer').then(m => ({ default: m.DatasetViewer })))
const LoraLibrary = lazy(() => import('./components/LoraLibrary').then(m => ({ default: m.LoraLibrary })))
const ModelsView = lazy(() => import('./components/ModelsView').then(m => ({ default: m.ModelsView })))

export type Tab = 'train' | 'generate' | 'outputs' | 'datasets' | 'library' | 'models'

const PAGE_TITLES: Record<Tab, string> = {
  train: 'Training',
  generate: 'Generate',
  outputs: 'Outputs',
  datasets: 'Datasets',
  library: 'LoRA Library',
  models: 'Models',
}

function App() {
  const searchString = useSearch()
  const [, navigate] = useLocation()
  const TABS = ['generate', 'outputs', 'datasets', 'train', 'library', 'models'] as const
  const params = new URLSearchParams(searchString)
  const tab: Tab = TABS.find((t) => t === params.get('tab')) ?? 'generate'
  const setTab = (next: Tab) => navigate(`/?tab=${next}`)

  // Sidebar collapsed state (persisted)
  const [sidebarCollapsed, setSidebarCollapsed] = useLocalStorage<boolean>(
    'modl:sidebar-collapsed',
    () => false,
  )

  const TabLoading = (
    <div className="flex items-center justify-center py-24">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
    </div>
  )

  const TabError = (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <p className="text-sm font-medium text-foreground">Something went wrong</p>
      <p className="mt-1 text-xs text-muted-foreground">This section encountered an error. Other tabs still work.</p>
      <button
        type="button"
        onClick={() => window.location.reload()}
        className="mt-3 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
      >
        Reload
      </button>
    </div>
  )

  return (
    <FormProvider>
      <div className="flex h-screen overflow-hidden bg-background">
        {/* Desktop sidebar */}
        <div className="hidden md:flex md:flex-col md:h-full">
          <AppSidebar
            activeTab={tab}
            onTabChange={setTab}
            collapsed={sidebarCollapsed}
            onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
          />
        </div>

        {/* Mobile nav (top bar + bottom bar + drawer) */}
        <MobileNav activeTab={tab} onTabChange={setTab} />

        {/* Main column */}
        <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
          {/* Desktop page header */}
          <header className="hidden md:flex h-14 shrink-0 items-center gap-4 border-b border-border/50 bg-[#09090e]/90 backdrop-blur px-6">
            <h1 className="text-base font-semibold tracking-tight text-foreground">
              {PAGE_TITLES[tab]}
            </h1>
          </header>

          {/* Scrollable content */}
          <main className="flex-1 overflow-y-auto bg-[#09090e]">
            {/* Tab: Train — lazy-mounted */}
            {tab === 'train' && (
              <div className="flex h-full flex-col pb-24 pt-16 md:pb-0 md:pt-0">
                <ErrorBoundary fallback={TabError}>
                  <Suspense fallback={TabLoading}>
                    <TrainingRuns />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}

            {/* Tab: Generate — always mounted (primary view, preserves SSE + session state) */}
            <div className={tab === 'generate' ? 'h-full' : 'hidden'}>
              <GenerateView />
            </div>

            {/* Tab: Outputs — lazy-mounted */}
            {tab === 'outputs' && (
              <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
                <ErrorBoundary fallback={TabError}>
                  <Suspense fallback={TabLoading}>
                    <OutputsGallery />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}

            {/* Tab: Datasets — lazy-mounted */}
            {tab === 'datasets' && (
              <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
                <ErrorBoundary fallback={TabError}>
                  <Suspense fallback={TabLoading}>
                    <DatasetViewer />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}

            {/* Tab: Library — lazy-mounted */}
            {tab === 'library' && (
              <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
                <ErrorBoundary fallback={TabError}>
                  <Suspense fallback={TabLoading}>
                    <LoraLibrary />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}

            {/* Tab: Models — lazy-mounted */}
            {tab === 'models' && (
              <div className="px-4 py-6 pb-24 md:px-6 md:pb-6 pt-16 md:pt-6">
                <ErrorBoundary fallback={TabError}>
                  <Suspense fallback={TabLoading}>
                    <ModelsView />
                  </Suspense>
                </ErrorBoundary>
              </div>
            )}
          </main>
        </div>

        {/* Floating generation queue panel — hidden on Generate tab (session strip handles it) */}
        {tab !== 'generate' && <QueuePanel />}
      </div>
    </FormProvider>
  )
}

export default App
