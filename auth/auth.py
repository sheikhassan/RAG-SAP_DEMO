"""User authentication: CSV-backed users, PBKDF2 hashing, JWT issue/verify.

Storage format (users.csv):
    username,password_hash,role,full_name,email

password_hash format:
    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>

Demo users are seeded on first import if `users.csv` is missing. In a real
deployment, manage `users.csv` outside the repo and rotate JWT_SECRET.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

import jwt

from src.config import (
    AUTH_DIR,
    JWT_ALGORITHM,
    JWT_SECRET,
    JWT_TTL_HOURS,
    ROLES,
    USERS_CSV,
)

_PBKDF2_ITERATIONS = 200_000
_PBKDF2_PREFIX = "pbkdf2_sha256"


class AuthError(Exception):
    """Raised for malformed credentials, expired tokens, etc."""


@dataclass
class User:
    username: str
    role: str
    full_name: str
    email: str


# --- Password hashing -----------------------------------------------------


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iters_s, salt_hex, hash_hex = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != _PBKDF2_PREFIX:
        return False
    try:
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(actual, expected)


# --- CSV store ------------------------------------------------------------

_CSV_FIELDS = ["username", "password_hash", "role", "full_name", "email"]

# Demo accounts. Documented in README. Change passwords for any real use.
_DEMO_USERS: list[tuple[str, str, str, str, str]] = [
    # username,    password,     role,         full_name,           email
    ("admin",      "admin123",   "admin",      "Site Admin",        "admin@drivemedical.local"),
    ("hr",         "hr123",      "hr",         "HR Coordinator",    "hr@drivemedical.local"),
    ("manager",    "manager123", "manager",    "Ops Manager",       "manager@drivemedical.local"),
    ("alice",      "alice123",   "finance",    "Alice (Finance)",   "alice@drivemedical.local"),
    ("bob",        "bob123",     "procurement","Bob (Procurement)", "bob@drivemedical.local"),
    ("carol",      "carol123",   "planning",   "Carol (Planning)",  "carol@drivemedical.local"),
]


def seed_users_if_missing() -> None:
    """Create users.csv with demo accounts if it doesn't exist."""
    if USERS_CSV.exists():
        return
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with USERS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for username, password, role, full_name, email in _DEMO_USERS:
            w.writerow(
                {
                    "username": username,
                    "password_hash": _hash_password(password),
                    "role": role,
                    "full_name": full_name,
                    "email": email,
                }
            )


def _load_user(username: str) -> Optional[dict]:
    if not USERS_CSV.exists():
        return None
    target = (username or "").strip().lower()
    if not target:
        return None
    with USERS_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("username") or "").strip().lower() == target:
                return row
    return None


# --- Public API -----------------------------------------------------------


def authenticate(username: str, password: str) -> Optional[User]:
    """Return a User on success, None on bad credentials."""
    seed_users_if_missing()
    row = _load_user(username)
    if not row:
        # Run a dummy hash compare to keep timing similar for missing users.
        _verify_password(password, _hash_password(secrets.token_hex(8)))
        return None

    if not _verify_password(password, row.get("password_hash", "")):
        return None

    role = (row.get("role") or "").strip().lower()
    if role not in ROLES:
        return None

    return User(
        username=row.get("username", "").strip(),
        role=role,
        full_name=(row.get("full_name") or "").strip(),
        email=(row.get("email") or "").strip(),
    )


def issue_token(user: User) -> str:
    now = int(time.time())
    payload = {
        "sub": user.username,
        "role": user.role,
        "name": user.full_name,
        "email": user.email,
        "iat": now,
        "exp": now + JWT_TTL_HOURS * 3600,
        "iss": "rag-sap-chatbot",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Return the decoded payload if valid, otherwise None."""
    if not token:
        return None
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

    if (payload.get("role") or "").strip().lower() not in ROLES:
        return None
    return payload


# --- Bootstrap on import (idempotent) -------------------------------------

# Create the demo CSV the first time the module is loaded so a fresh clone of
# the repo is immediately usable. Safe to call repeatedly.
try:
    seed_users_if_missing()
except OSError:
    # Filesystem may be read-only in some sandboxes; ignore at import time.
    pass

# Soft warning if running with the dev secret in non-dev env.
if JWT_SECRET == "dev-secret-change-me" and os.getenv("ENV") == "production":  # pragma: no cover
    raise RuntimeError("JWT_SECRET must be set to a strong value in production")
