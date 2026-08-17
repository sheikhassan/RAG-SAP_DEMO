---
title: SIOP Monthly Cycle
role: planning
system: both
source_id: PLAN-SIOP-001
---

# SIOP (Sales, Inventory & Operations Planning) — Drive Medical

The SIOP process aligns demand, supply, inventory, and financial plans on a
rolling 18-month horizon. It runs monthly with five gated steps.

## SIOP Calendar (Workdays)
| Day | Step                       | Owner               |
|-----|----------------------------|---------------------|
| WD-1 to WD+2 | Data refresh         | Planning ops        |
| WD+3 | Demand Review              | Demand Manager      |
| WD+5 | Supply Review              | Supply Planner      |
| WD+7 | Inventory & Reconciliation | Inventory Analyst   |
| WD+9 | Pre-SIOP                   | Director Planning   |
| WD+11| Executive SIOP             | VP Operations       |

## Step 1 — Demand Review
- Inputs: published IBP Consensus Demand (`CONSENSUSDEMAND`), prior month
  bias report, NPI list.
- Output: Approved demand plan locked in IBP scenario `BASELINE`.

## Step 2 — Supply Review
- Inputs: approved demand, current inventory snapshot, capacity by line.
- Activities:
  1. Run rough-cut capacity (RCCP) in IBP for Response.
  2. Identify constrained periods (load > 95% of available capacity).
  3. Propose mitigations: overtime, alternate plants, or demand shaping.
- Output: Constrained supply plan in scenario `SUPPLY-FEASIBLE`.

## Step 3 — Inventory & Reconciliation
- Reconcile target vs projected ending inventory by product family.
- Validate days-of-supply against Drive Medical policy (target = 45 days for
  finished goods, 30 days for components).

## Step 4 — Pre-SIOP
- Surface gaps and decisions needed at executive level (e.g., capex requests,
  inter-plant transfers, customer allocation).
- Director Planning chairs; outputs the executive deck.

## Step 5 — Executive SIOP
- VP Operations chairs. Decisions are minuted and assigned in the SIOP
  decision log (SharePoint list `SIOP-Decisions`).
- Approved plan becomes the Drive Medical operating plan for the next 90 days.

## Key Reports
- `SIOP_GAP_REPORT` — IBP analytic showing demand vs supply by family.
- `SIOP_DOS_TREND` — Days-of-supply trend by family.
- `SIOP_FINANCIAL_VIEW` — converts the volume plan to revenue at standard
  prices for finance reconciliation.
