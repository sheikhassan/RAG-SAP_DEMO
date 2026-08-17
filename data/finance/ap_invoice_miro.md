---
title: AP Invoice Processing using MIRO
role: finance
system: s4hana
source_id: FI-AP-001
---

# AP Invoice Processing (MIRO) — Drive Medical S/4HANA

This procedure describes how Accounts Payable users post a vendor invoice against
a purchase order in SAP S/4HANA at Drive Medical.

## Prerequisites
- A released Purchase Order (PO) exists in the system.
- Goods Receipt (GR) has been posted by the warehouse team via MIGO.
- The original vendor invoice (PDF) is available in the AP shared mailbox.

## Step-by-Step Procedure
1. Log in to SAP Fiori Launchpad with your AP role.
2. Open the **Supplier Invoice** app (transaction code `MIRO`).
3. In the header, set:
   - **Transaction**: Invoice
   - **Invoice Date**: date printed on the supplier invoice
   - **Reference**: vendor invoice number (mandatory at Drive Medical)
   - **Amount**: gross amount including tax
   - **Tax Code**: select per Drive Medical tax matrix (e.g. `V0` for non-taxable)
4. In the **PO Reference** tab, enter the PO number. The system auto-populates
   line items from the open GR.
5. Verify quantities and amounts match the PO and GR. If there is a price
   variance greater than 5%, the system blocks the invoice for payment — this is
   intentional per Drive Medical AP policy.
6. Attach the scanned vendor invoice PDF using the paperclip icon
   (Services for Object → Create → Attachment).
7. Click **Simulate** to preview the accounting entries.
8. Click **Post**. Note the document number (starts with `51` for AP invoices).

## Common Issues
- **"Balance not zero"** — usually a tax code mismatch. Cross-check with the PO.
- **"GR/IR account not cleared"** — wait 15 minutes for the GR posting to
  replicate, then retry.
- **Invoice blocked for payment** — open `MRBR` to review and release after
  approval from the cost center owner.

## Approval Workflow
After posting, invoices above $10,000 trigger a Drive Medical workflow to the
respective cost center owner for release before F110 picks them up.
