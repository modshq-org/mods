# Session Strip — Unified Queue + Output Timeline

## Problem

Three disconnected surfaces on the Generate tab:

1. **Floating queue pill** (bottom-right) — shows "generating..." but results vanish
   when done. No connection to where outputs went.
2. **Canvas** — only shows results if SSE "completed" fires while you're looking at it.
   Queue 5 jobs, browse Outputs tab, come back — canvas is empty.
3. **History filmstrip** (bottom of canvas) — polls `/api/outputs` every 15s. No queue
   awareness. Shows all-time history, not the current session.

The user queues 5 generations, goes to check models, comes back, and has no idea what
happened. The pill is gone, the canvas is blank, the filmstrip maybe refreshed.

## Design: Merge Queue + Filmstrip into Session Strip

Replace the filmstrip with a **session-aware timeline strip** at the bottom of the
Generate canvas. Each generation job in the current session gets a card:

```
[✓ thumb] [✓ thumb] [▶ 64%] [○ queued] [○ queued]    | history →
```

### Card States

| State     | Visual                                           | Interaction          |
|-----------|--------------------------------------------------|----------------------|
| Queued    | Dimmed card, model dot, prompt snippet           | X to cancel          |
| Active    | Skeleton/partial + progress bar overlay           | Canvas auto-tracks   |
| Completed | Thumbnail + model dot                            | Click → canvas shows |
| Error     | Red tint + warning icon                          | Click → shows error  |

### Canvas Behavior

- **Auto-advance**: When a job completes, canvas shows its result.
- **User focus**: If user clicked an older card (reviewing), don't auto-advance.
  Show a subtle "New result →" nudge instead.
- **Multi-image batches**: A card can expand to show N thumbnails for batch_count > 1.

### Floating Pill

- **Hidden on Generate tab** — the strip handles queue visualization.
- **Visible on other tabs** — current behavior. Clicking navigates to Generate tab
  and the strip shows the full picture.

### Data Source

The strip maintains a **session array** in React state (not localStorage — ephemeral):

```typescript
type SessionItem = {
  id: string                    // unique key
  status: 'queued' | 'active' | 'completed' | 'error'
  prompt: string
  model_id: string
  batch_count: number
  // Populated on completion:
  images?: PreviewImage[]
  error?: string
  // Progress (from SSE):
  step?: number
  totalSteps?: number
}
```

Items are added when you hit Generate (locally, before server responds).
Status updates come from SSE events. Completed images come from outputs refresh.

On page refresh, the session array resets (it's ephemeral). The filmstrip falls
back to recent history (current behavior) until new jobs are submitted.

### Divider: Session vs History

After the session items, a subtle divider separates "this session" from
"recent history" (pulled from `/api/outputs` as today). This keeps the familiar
filmstrip behavior for browsing past work.

```
[session items...] | [recent history thumbnails...]
```

## Implementation Notes

### SSE Events Needed

The current SSE already sends:
- `queue:N` — queue count changed
- `completed` / `done` — job finished
- `{"step": N, "total_steps": M}` — progress

Additional event needed:
- `job:start:{"prompt":"...","model_id":"..."}` — so the strip knows which
  session item is now active (currently the strip would have to guess based on
  order, which is fine for v1).

### Queue Details API

`GET /api/generate/queue` already returns `current` and `queue` with prompt/model.
The strip can poll this for initial sync (e.g., page refresh while jobs are running).

### Migration Path

1. Add `sessionItems` state to GenerateView
2. Populate on handleGenerate/handleEdit
3. Update from SSE events
4. Replace `<GenerationGallery>` with `<SessionStrip>` that renders session items
   first, then falls back to history
5. Hide `<QueuePanel>` pill when on Generate tab
