"""Production clock and identifier boundary implementations."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class SecureIdGenerator:
    def new_id(self) -> str:
        return secrets.token_hex(5)
