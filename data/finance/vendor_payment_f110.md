---
title: Vendor Payment Run (F110)
role: finance
system: both
source_id: FI-AP-002
---

# Vendor Payment Run (F110) — Drive Medical

The Automatic Payment Program (`F110`) is run twice weekly by AP at Drive Medical
to pay open vendor invoices. The procedure below works on both ECC and S/4HANA.

## Schedule
- **Tuesday 10:00 AM CST** — domestic ACH run
- **Friday 10:00 AM CST** — international wire run

## Step-by-Step
1. Run transaction `F110`.
2. Enter:
   - **Run Date**: today
   - **Identification**: `DM01` for domestic ACH, `DMWI` for wires
3. On the **Parameters** tab:
   - **Posting Date**: today
   - **Docs entered up to**: today
   - **Company Code**: `DM01`
   - **Payment Method**: `A` (ACH) or `W` (Wire)
   - **Vendor range**: leave blank for all, or restrict per business request
4. Save parameters.
5. Click **Proposal** → schedule immediately. Wait for status
   *"Payment proposal has been created"*.
6. Click **Edit Proposal** to review. Exclude any invoices flagged by Treasury
   (right-click → Block).
7. Click **Payment Run** → schedule immediately.
8. After completion, click **Printout/Data Medium** to generate the ACH file
   (`PMW` format). The file lands in `\\dm-bank-share\outbound\` and is picked
   up by the bank within 30 minutes.

## Reconciliation
- Run `FBL1N` for the affected vendors and confirm the cleared status.
- Treasury validates the bank confirmation file by 4 PM the same day.

## Troubleshooting
- **"No valid payment method"** — vendor master `XK02` is missing payment method
  in the Company Code data. Contact the Vendor Master team.
- **Run stuck in "Payment run is running"** — check job `RFFOAVIS_FPAYM` in
  `SM37`; if it errored, restart the print step only.
