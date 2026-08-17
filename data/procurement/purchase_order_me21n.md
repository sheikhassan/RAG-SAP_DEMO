---
title: Purchase Order Creation (ME21N)
role: procurement
system: s4hana
source_id: PROC-PO-001
---

# Purchase Order Creation (ME21N) — S/4HANA

Procurement buyers at Drive Medical create Purchase Orders using transaction
`ME21N` (or the **Manage Purchase Orders** Fiori app in S/4HANA).

## Prerequisites
- Approved Purchase Requisition (PR) exists, OR you have a Source List entry
  for the material/vendor combination.
- Vendor master is fully maintained (purchasing org `1000`, Drive Medical).
- Material master has the **Purchasing** view extended.

## Step-by-Step
1. Open the **Manage Purchase Orders** app or run `ME21N`.
2. Select **Document Type**:
   - `NB` — Standard PO
   - `FO` — Framework / Blanket order (annual contracts)
   - `UB` — Stock transport order (between DM warehouses)
3. In the **Header** section enter:
   - **Vendor**: the supplier number (or search by name)
   - **Purchasing Org**: `1000`
   - **Purchasing Group**: your buyer code (e.g. `B01`)
   - **Company Code**: `DM01`
4. In the **Item Overview** grid add each line:
   - **Material** number (or short text for non-stock)
   - **Quantity** and **UoM**
   - **Net Price**
   - **Plant** (where the goods are received)
   - **Storage Location**
   - **Delivery Date**
5. Open the **Item Detail** for each line and confirm the **Account Assignment
   Category** — `K` for cost center, `P` for project, blank for stock.
6. Click **Check** (Ctrl+F2). Resolve any red errors.
7. Click **Save** (Ctrl+S). The system assigns a PO number starting with
   `45XXXXXXXX`.

## Approval (Release Strategy)
At Drive Medical, POs follow this release strategy:
- Up to **$5,000** — auto-released, no approval.
- **$5,000 – $25,000** — Buyer Manager release (`ME29N`, code `B1`).
- **$25,000 – $100,000** — Director release (code `D1`).
- **Above $100,000** — VP Procurement release (code `V1`).

## Outputting to Vendor
After full release, the PO is automatically emailed to the vendor's contact
maintained in `XK03` → Purchasing data → Email. To resend manually, run `ME9F`
with the PO number.
