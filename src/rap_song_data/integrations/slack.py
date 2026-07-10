"""Optional Slack notifications for model jobs."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def slack_webhook_from_env() -> str:
    return os.environ.get("SLACK_WEBHOOK_URL", "").strip()


def notify_slack(webhook_url: str | None, text: str, fields: dict[str, Any] | None = None) -> bool:
    if not webhook_url:
        return False

    if fields:
        detail_lines = [f"*{key}:* {value}" for key, value in fields.items() if value is not None]
        if detail_lines:
            text = text + "\n" + "\n".join(detail_lines)

    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
