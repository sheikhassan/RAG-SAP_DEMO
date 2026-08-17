---
title: Purchase Requisition Creation and Approval
role: procurement
system: both
source_id: PROC-PR-001
---

# Purchase Requisition (PR) — Drive Medical

A Purchase Requisition is the internal request for procurement to buy goods or
services. It precedes the Purchase Order and is governed by Drive Medical's
release strategy.

## Creating a PR (`ME51N`)
1. Run `ME51N` or open the **Create Purchase Requisition** Fiori app.
2. Choose **Document Type**:
   - `NB` — Standard
   - `RV` — Stock transfer
3. For each line item enter:
   - **Material** (or short text + material group for non-catalog)
   - **Quantity**, **UoM**
   - **Delivery Date** (must be ≥ 5 business days out for standard items)
   - **Plant** and **Storage Location**
   - **Account Assignment**: `K` (cost center) or `P` (project)
4. In **Item Detail → Source of Supply** select the preferred vendor from the
   info record (or leave blank for buyer to source).
5. Save. The PR number starts with `10XXXXXXXX`.

## Approval Workflow
PRs at Drive Medical follow this approval matrix (release group `DM`):

| Value Band            | Approver Role         | Code |
|-----------------------|-----------------------|------|
| Up to $1,000          | Cost center owner     | C1   |
| $1,000 – $10,000      | Department Manager    | D1   |
| $10,000 – $50,000     | Director              | E1   |
| Above $50,000         | VP + Finance review   | V1   |

Approvers receive a workflow item in their **My Inbox** Fiori app and can
release using `ME54N` or directly from the inbox.

## Tracking PR Status
- Run `ME53N` and enter the PR number.
- Header status `B` = released, ready for PO conversion.
- Status `N` = not yet released.
- Use report `ME5A` to list all your open PRs.

## Common Issues
- **PR fails release** — usually due to missing cost center or invalid GL
  account. Check Item → Account Assignment.
- **No source determined** — buyer needs to maintain an info record in `ME11`.
