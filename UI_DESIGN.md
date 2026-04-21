# MEDUSA UI Design

## Intent

The custom MEDUSA interface should feel like a data operations control room, not a generic CRUD dashboard.
The product is an RL environment, but the UI should make it legible to humans:

- choose a task from the deterministic benchmark catalog
- select a built-in MEDUSA agent
- run the full trace automatically
- inspect how Bronze, prepped, joined, Silver, and quarantine tables change
- understand why a trace is healthy or risky
- see how the final grader interprets the run

The frontend is intentionally no-session. The browser owns `task_id + agent_id + actions[]`, and every server call replays that trace.

## Information Architecture

Primary routes:

- `/medusa/tasks`: task catalog
- `/medusa/studio`: episode studio
- `/medusa/audit`: audit report
- `/medusa`: redirects to `/medusa/studio`

Supporting API routes:

- `GET /api/tasks`
- `GET /api/tasks/{task_id}`
- `GET /api/action-space`
- `GET /api/agents`
- `POST /api/run/preview`
- `POST /api/run/reset/{task_id}`
- `POST /api/run/step`
- `POST /api/run/autorun/{task_id}`
- `POST /api/run/tables`
- `POST /api/run/timeline`
- `POST /api/run/analysis`
- `POST /api/run/feature-vector`
- `POST /api/run/grader`
- `POST /api/run/evaluate`

## Layout

### 1. Task Catalog

Purpose:

- let users browse deterministic tasks before touching the simulator
- explain difficulty, seed, success criteria, and rubric at a glance
- act as the clean entry point for the product

### 2. Episode Studio

Purpose:

- provide the main automated replay surface
- visualize current episode state, tables, and the governance timeline
- present the interface as the Medusa Live Playground

### 3. Audit Report

Purpose:

- isolate grader output and rubric interpretation into a focused review page
- make feature-vector understanding part of the evaluation experience
- support “did this trace really work?” validation after experimentation

### 4. Shared Hero

Purpose:

- frame the app as a MEDUSA control room
- reinforce that the UI is replay-driven and deterministic
- show current run status at a glance

### 5. Scenario Rail

Left column.

Contents:

- task picker
- agent picker
- rerun control
- task description and success criteria
- current action trace

Why:

- users need a stable place to understand the target task before acting
- the trace is still the core artifact, but it is now produced by a selected agent rather than manual clicks

### 6. Episode Overview

Center column, top.

Contents:

- summary metrics
- latest observation message
- commit-readiness diagnostics
- selected agent strategy card

Why:

- this is the decision surface
- it should tell the user what the selected agent just did, what is risky, and why the run is or is not safe

### 7. Table Explorer

Wide lower-middle surface.

Contents:

- table selector
- page controls
- row preview

Why:

- MEDUSA is fundamentally about pipeline state transitions
- users need direct visibility into Bronze, joined, Silver, and quarantine states

### 8. Timeline

Right-side narrative panel.

Contents:

- governance log entries
- per-step reward
- key metrics emitted by operators

Why:

- this is the explainability spine for the environment
- it helps users connect action choices to reward outcomes

### 9. Audit Surface

Bottom band.

Contents:

- grader report
- rubric breakdown
- feature-vector decode

Why:

- a MEDUSA run is only meaningful once it can be audited
- the feature vector helps bridge the gap between human understanding and RL observations

## Visual Direction

The visual tone should suggest “data foundry” more than “enterprise admin”.

Guidelines:

- warm parchment base instead of flat white
- copper and teal accents to imply transformation and governance
- serif headlines for weight and identity
- translucent panels with subtle depth so the studio feels layered
- a visible grid texture in the background to hint at tabular data without becoming noisy

## Interaction Model

The interaction loop is:

1. Select task
2. Select agent
3. Run `POST /api/run/reset/{task_id}`
4. Run `POST /api/run/step` repeatedly until the selected agent reaches `done`
5. Store the returned `actions[]` in the browser
6. Refresh dependent views from replay endpoints

No server-side lifecycle needs to be maintained.

## Frontend State

The browser should store:

- `taskId`
- `agentId`
- `actions[]`
- `selectedTable`
- `tablePage`
- most recent preview payload

Everything else is derived from replay responses.

## UX Priorities

- keep the action consequences visible
- make table changes easy to inspect
- surface grader readiness before commit
- make failure states informative rather than punitive
- avoid requiring users to understand OpenEnv internals

## Future Extensions

- side-by-side trace compare
- import/export trace JSON
- seeded scenario mode beyond catalog tasks
- heuristic recommendation panel
- sparkline history for reward and match-rate changes
