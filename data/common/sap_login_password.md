---
title: SAP Login and Password Reset
role: common
system: both
source_id: COMMON-LOGIN-001
---

# SAP Login & Password Reset — Drive Medical

This applies to all Drive Medical SAP users regardless of role.

## Logging In
1. Open the **SAP Fiori Launchpad** at `https://fiori.drivemedical.com`
   (S/4HANA), or launch **SAP Logon** for ECC and double-click `DMP` (PRD).
2. Enter:
   - **Client**: `100` (Production) or `200` (QA — for testing only)
   - **User**: your Drive Medical SSO ID (e.g. `JSMITH`)
   - **Password**: your current SAP password
   - **Language**: `EN`
3. On first login of the day you will be prompted to confirm your role
   assignment in S/4HANA — click **OK**.

## Forgotten Password
1. Open `https://sap-self-service.drivemedical.com`.
2. Click **Reset SAP Password**.
3. Authenticate with your Windows / Azure AD credentials.
4. The portal generates a temporary password and emails it to your Drive
   Medical address.
5. Log in with the temp password — you will be forced to change it
   immediately.

## Account Locked
After **5** failed attempts your SAP user is locked. To unlock:
- Wait 30 minutes — the lock auto-clears, OR
- Raise an IT ticket of category **SAP → Account Unlock**. Service desk
  unlocks the user in `SU01` within the SLA (15 minutes during business
  hours).

## Multi-Factor Authentication
S/4HANA Fiori requires MFA via Microsoft Authenticator. ECC SAP GUI is on
the Drive Medical VPN, so no in-app MFA is required, but VPN itself enforces
MFA at connect time.
