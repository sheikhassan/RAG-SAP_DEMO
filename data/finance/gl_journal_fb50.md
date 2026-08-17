---
title: General Ledger Journal Entry (FB50)
role: finance
system: ecc
source_id: FI-GL-001
---

# General Ledger Journal Entry (FB50) — ECC

GL accountants at Drive Medical use `FB50` to post manual journal entries in
SAP ECC for accruals, reclassifications, and month-end adjustments.

## When to Use FB50
- Monthly accrual entries (e.g. utilities, professional services).
- Reclassification between cost centers.
- Correction of a prior-period AP/AR misposting (with controller approval).

## Procedure
1. Run `FB50`.
2. Enter the header:
   - **Document Date**: original transaction date
   - **Posting Date**: current period
   - **Reference**: short description (max 16 chars), e.g. `ACCR-NOV-UTIL`
   - **Doc Header Text**: business reason
3. In the line item grid, for each line enter:
   - **G/L Account**
   - **D/C** (Debit or Credit)
   - **Amount in doc. curr.**
   - **Cost Center** (mandatory for P&L accounts at Drive Medical)
   - **Text** (line-level description)
4. Click **Simulate** (Shift+F9) to preview.
5. Confirm **Total Dr = Total Cr** in the status bar.
6. Click **Post** (Ctrl+S). Note the document number.

## Drive Medical Specifics
- All accruals must reference a cost center; the system rejects P&L lines
  without one.
- Manual JEs above $50,000 require a second-level approval entry in the
  internal SOX log before posting.
- Inter-company JEs use `FB50L` with company code `DM01` ↔ `DM02`.

## Reversal
To reverse an erroneous entry, use `FB08` with reversal reason `01` (reversal
in current period) or `02` (reversal in prior period — controller approval
required).
