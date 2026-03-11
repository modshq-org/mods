# Generate Tab & Queue Revamp — Code Review

Review of the queue/loading/model/LoRA UI rework on `feat/model-families`.

## HIGH severity

### 1. Missing dependency array on keyboard shortcut effect
**GenerateView.tsx:291** — The `useEffect` for Ctrl+Enter has no dependency array, re-registering the listener on every render. Creates/tears down an event listener every render cycle.

**Fix:** Add deps array, or use a ref for form values + stable handler with `[]` deps.

### 2. Direct mutation of form state
**GenerateView.tsx:316, 426** — Both `handleGenerate` and `handleEdit` mutate `form.seed` directly (`form.seed = newSeed`). This bypasses React's immutability guarantees.

**Fix:** Use the `newSeed` variable directly in the request construction instead of mutating `form.seed`. The `setForm` call already handles state update.

### 3. `handleCancelQueued` stale closure causes cascading re-renders
**GenerateView.tsx:606** — Depends on `[sessionItems]`, so it's re-created on every session item change (including progress ticks). All downstream components (SessionStrip, GenerateActions) re-render on every tick.

**Fix:** Use a ref for sessionItems: `sessionItemsRef.current = sessionItems` and read from ref inside callback with `[]` deps.

### 4. SSE connection never closed on successful completion
**GenerateView.tsx** — `setSseConnected(true)` is called on submit but `setSseConnected(false)` is never called on successful completion. The EventSource stays open indefinitely after the first generation.

**Fix:** Call `setSseConnected(false)` in the completion handler when no more queued items remain.

## MEDIUM severity

### 5. ~100 lines duplicated between handleGenerate and handleEdit
**GenerateView.tsx:309-415 vs 418-517** — Seed randomization, session item creation, SSE connection, error handling, queue parsing, and toast notifications are copy-pasted. Only request body construction differs.

**Fix:** Extract shared submission orchestration into a `submitJob()` function.

### 6. sendToEdit duplicates model-switching logic from handleModeSwitch
**GenerateView.tsx:558-589 vs 233-285** — Edit model selection (find qwen-image-edit, apply defaults) is duplicated.

**Fix:** Extract into a shared helper.

### 7. handleGenerate/handleEdit dependency arrays too broad
**GenerateView.tsx:415, 517** — Depend on `[form, isGenerating]`. `form` changes on every keystroke, re-creating these callbacks and propagating re-renders.

**Fix:** Read `form` from a ref, or destructure only needed fields.

### 8. Multiple setForm calls in handleGallerySelect
**GenerateView.tsx:529-536** — Up to 6 sequential `setForm` calls. Each creates a new object.

**Fix:** Combine into a single `setForm` call with conditional spreads.

### 9. Inline `<style>` tag for keyframes in ImagePreview
**ImagePreview.tsx:202-211** — CSS keyframes injected via inline `<style>` on every render of the generating state.

**Fix:** Move `@keyframes gradientShift` and `@keyframes shimmer` to `index.css`.

### 10. Collapsed training indicator mispositioned
**AppSidebar.tsx:123** — `absolute`-positioned dot inside a `<button>` that lacks `relative`. The dot positions relative to the `<aside>`, not the button.

**Fix:** Add `relative` to the nav button's className.

### 11. Duplicate GPU query in 3 components
**App.tsx, AppSidebar.tsx, GenerateView.tsx** — Same `['gpu']` query with identical config defined independently. Could drift.

**Fix:** Extract into a `useGpuStatus()` hook.

### 12. Queue items keyed by array index
**QueuePanel.tsx:160** — `key={i}` on queue items. When items are removed, React reconciles incorrectly.

**Fix:** Use a composite key or add IDs to the server queue response.

### 13. `queuePosition` logic misleading
**GenerateView.tsx:627-628** — `queuedBefore` counts items with status `'queued'`, but these are items *behind* the active one. The variable name and logic are confusing. During `submitting`, it may incorrectly show "In queue" for the active job.

**Fix:** Queue position for the active item should always be 0. Use the server's queue length response instead.

### 14. `isLoraCompatible` returns true when both families are undefined
**LoraPanel.tsx:52** — If both `loraParent` and `selectedParent` are undefined, `undefined === undefined` is `true`, incorrectly marking the LoRA as compatible.

**Fix:** Guard: `if (!loraParent || !selectedParent) return true` before comparison.

### 15. All tabs always mounted
**App.tsx:86-115** — Every tab is rendered and hidden with CSS. DatasetViewer, ModelsView, etc. run their queries even when not visible.

**Fix:** Conditionally render heavy tabs. Keep GenerateView always-mounted to preserve state.

### 16. Accessibility — icon-only buttons lack aria-labels
**Multiple files** — Cancel/close buttons (ImagePreview, GenerateActions, SessionStrip, QueuePanel) have no `aria-label`. Screen readers announce them as unlabeled.

**Fix:** Add `aria-label` to all icon-only buttons.

## LOW severity

### 17. Division by zero in GenerateProgressBar
**GenerateProgressBar.tsx:58** — Doesn't guard `totalSteps > 0`. If 0, produces `Infinity%` CSS width.

### 18. Double model lookup in Img2ImgPanel prop
**GenerateView.tsx:729-732** — `models.find(...)` called twice with the same predicate inline.

### 19. Hardcoded dark theme hex colors
**Multiple files** — `#0e0e18`, `#1a1030`, etc. scattered across components. Won't adapt to theme changes.

### 20. Unused `_setTab` prop
**GenerateView.tsx:38** — Destructured but never used. Remove or wire up.

### 21. Inconsistent null safety on form.edit_images
**GenerateActions.tsx:38** — Uses `?.` optional chaining on a field typed as non-optional. Harmless but signals confusion about the type contract.

## Refactoring priorities (bang for buck)

1. **Extract `submitJob()` helper** — Eliminates ~100 lines of duplication, reduces bug surface (items 5, 6)
2. **Fix re-render cascade** — Ref-ify sessionItems in callbacks, narrow `form` deps (items 3, 7)
3. **Move keyframes to CSS** — Quick cleanup (item 9)
4. **Close SSE on completion** — Prevents connection leak (item 4)
5. **Extract `useGpuStatus()` hook** — Centralizes query config (item 11)
