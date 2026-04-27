import json
from datetime import datetime
from pathlib import Path


def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_ts(ts):
    return datetime.fromisoformat(ts)


def build_user_timeline(user):
    convs = sorted(user["conversations"], key=lambda x: x["timestamp"])
    first = parse_ts(convs[0]["timestamp"])
    timeline = []
    for idx, conv in enumerate(convs, start=1):
        ts = parse_ts(conv["timestamp"])
        week = int(((ts - first).days // 7) + 1)
        timeline.append(
            {
                "session_number": idx,
                "session_id": conv.get("session_id"),
                "timestamp": ts,
                "week": week,
                "tags": conv.get("tags", []),
                "user_message": conv.get("user_message", ""),
                "user_followup": conv.get("user_followup", ""),
            }
        )
    return timeline

