---
title: IBP Demand Planning Cycle
role: planning
system: s4hana
source_id: PLAN-IBP-001
---

# IBP Demand Planning — Drive Medical

Drive Medical runs SAP Integrated Business Planning (IBP) for Demand on a
weekly cadence. The output of this cycle is the **Consensus Demand Plan**
(key figure `CONSENSUSDEMAND`) that feeds the SIOP process and downstream MRP.

## Weekly Cycle Calendar
- **Monday** — Statistical forecast run (automated overnight Sunday).
- **Tuesday** — Demand planner review and override.
- **Wednesday** — Sales input meeting (Sales Adjusted Demand uploaded).
- **Thursday** — Marketing input (promotions and NPI).
- **Friday morning** — Consensus meeting and plan freeze.
- **Friday afternoon** — Publish to S/4HANA via the IBP–S/4 integration job.

## Reviewing the Statistical Forecast
1. Open the **IBP Excel Add-in** and connect to planning area `DMDEMAND`.
2. Open the **Demand Planner Review** template.
3. Filter to your Product Family + Region.
4. Compare key figures:
   - `STATISTICALFORECASTQTY` — model output
   - `SALESACTUALSQTY` — last 12 months actuals
   - `MAPE` — current forecast accuracy
5. For SKUs with MAPE > 30%, click **Forecast Model Details** to see which
   model the system selected (Croston, Holt-Winters, etc.) and adjust if
   needed.

## Posting Demand Overrides
1. Enter your override in the `DEMANDPLANNERADJUSTMENT` key figure.
2. Add a comment justifying the change (mandatory at Drive Medical for any
   adjustment greater than ±15% of statistical).
3. Save the worksheet to commit to the planning area.

## Consensus Meeting
The Friday meeting reviews the `CONSENSUSDEMAND` view. Once agreed, the demand
manager runs the **Snapshot** application job to lock the plan. The published
plan flows into S/4 as a Planned Independent Requirement (PIR).

## Troubleshooting
- **Excel add-in disconnects** — switch to the IBP web client and refresh from
  there; reconnect Excel.
- **Statistical run failed** — check application job log `IBP_FORECAST_RUN`
  in the IBP web UI. Most failures are due to insufficient history (< 12
  months) on new SKUs.
