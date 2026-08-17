"""Authentication and authorization for the RAG SAP chatbot.

Public API:
    authenticate(username, password) -> User | None
    issue_token(user) -> str
    verify_token(token) -> dict | None
    seed_users_if_missing() -> None
"""
from .auth import (
    AuthError,
    User,
    authenticate,
    issue_token,
    seed_users_if_missing,
    verify_token,
)

__all__ = [
    "AuthError",
    "User",
    "authenticate",
    "issue_token",
    "seed_users_if_missing",
    "verify_token",
]
