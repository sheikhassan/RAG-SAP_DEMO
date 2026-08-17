---
title: MRP Run (MD01) Procedure
role: planning
system: ecc
source_id: PLAN-MRP-001
---

# MRP Run (MD01) — Drive Medical ECC

The Material Requirements Planning (MRP) run translates demand (sales orders +
PIRs) into supply proposals: planned orders for in-house production and
purchase requisitions for procurement.

## Run Schedule
- **Nightly run** — automated job `DM_MRP_NIGHTLY` runs at 02:00 CST for all
  plants (`1000`, `2000`, `3000`).
- **Manual ad-hoc** — planners may run `MD03` (single-material) or `MD02`
  (single-level) during the day after a major change.

## Running MRP for a Plant (`MD01`)
1. Run `MD01`.
2. Enter:
   - **Plant**: e.g. `1000`
   - **Processing Key**: `NETCH` (net change — only changed materials)
   - **Create Purchase Req**: `1` (purchase requisitions)
   - **Schedule Lines**: `3` (schedule lines for scheduling agreements)
   - **Create MRP List**: `1` (always create at Drive Medical for audit)
   - **Planning Mode**: `1` (adapt planning data)
   - **Scheduling**: `1` (basic dates, lead-time scheduling)
3. Press Enter, confirm the warning, and execute.
4. Wait for completion (typical 8–12 minutes for plant `1000`).

## Reviewing Results
- Use `MD05` (MRP list) for a snapshot at the time of the run.
- Use `MD04` (stock/requirements list) for the live view including changes
  since the run.
- Open exception messages from `MD06` filtered by your planner code.

## Common Exceptions
- **`07` — Postpone process order** — supply is too early; reschedule.
- **`10` — Reschedule in** — supply is too late; expedite or split the order.
- **`64` — Production finish after order finish** — capacity overload, resolve
  via `CM01` capacity planning.

## Conversion of Planned Orders
- Planned orders for in-house items: convert to process/production orders
  using `CO40` (mass) or `MD04` → right-click → Convert.
- Planned orders for purchased items: auto-converted to PRs by the run when
  the indicator is `1`. Buyer then converts PRs to POs in `ME21N`.
