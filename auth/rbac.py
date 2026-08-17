"""Role-based access helpers for the chat / upload UI.

This module is a thin convenience layer over `src.config` so UI code reads
naturally (`require_login(...)`, `require_upload_permission(...)`).
"""
from __future__ import annotations

from typing import Optional

from src.config import (
    can_upload as _can_upload,
    upload_targets as _upload_targets,
    visible_doc_roles as _visible_doc_roles,
)


class PermissionDenied(Exception):
    pass


def visible_doc_roles(user_role: str) -> list[str]:
    return _visible_doc_roles(user_role)


def upload_targets(user_role: str) -> list[str]:
    return _upload_targets(user_role)


def can_upload(user_role: Optional[str]) -> bool:
    return bool(user_role) and _can_upload(user_role)


def require_upload_permission(user_role: Optional[str], target_role: str) -> None:
    if not can_upload(user_role):
        raise PermissionDenied(
            f"Role '{user_role}' is not allowed to upload documents."
        )
    if target_role not in upload_targets(user_role):
        raise PermissionDenied(
            f"Role '{user_role}' is not allowed to upload to category '{target_role}'."
        )
