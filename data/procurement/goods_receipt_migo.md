---
title: Goods Receipt against PO (MIGO)
role: procurement
system: both
source_id: PROC-GR-001
---

# Goods Receipt — MIGO

Warehouse and procurement teams post Goods Receipts against open Purchase
Orders using `MIGO`. This works identically in ECC and S/4HANA.

## Prerequisites
- Released Purchase Order exists.
- Physical goods have arrived at the receiving dock and have been counted.
- Packing slip / delivery note from the carrier is available.

## Step-by-Step
1. Run `MIGO`.
2. In the top dropdown choose **Goods Receipt** + **Purchase Order**.
3. Enter the PO number, press Enter. The system loads all open lines.
4. Header tab:
   - **Document Date**: date on the packing slip
   - **Posting Date**: today
   - **Delivery Note**: carrier tracking or DN number
   - **Bill of Lading**: BOL number (mandatory at Drive Medical for audit)
5. For each line:
   - Set the **Item OK** checkbox.
   - Adjust **Qty in UnE** if you received less than ordered.
   - Confirm **Storage Location**.
   - If goods are damaged, set **Stock Type** to `Q` (quality inspection)
     instead of `F` (unrestricted).
6. Click **Check** then **Post**.
7. Note the material document number (starts with `50XXXXXXXX`). Print the
   GR slip (`Output → WE03`) and attach it to the physical packing slip.

## Partial Receipts
If only part of the order arrived, post the received quantity now. The PO
remains open for the balance, and a second GR can be posted later when the
rest arrives.

## Reversing a GR
Use `MIGO` → **Cancellation** → **Material Document**, enter the GR number,
and post. This is only allowed before the AP team has posted the invoice
(MIRO). After invoicing, contact AP for the proper reversal flow.

## Quality Inspection Releases
Goods posted to stock type `Q` are blocked until QA releases them via `QA32`.
Until released, they are visible in MMBE but cannot be issued to production.
