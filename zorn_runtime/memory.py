from __future__ import annotations

import os


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max(1, int(max_history))
        self._items = []

    def add(self, role: str, text: str):
        self._items.append({
            "role": role,
            "content": [{"type": "text", "text": str(text)}],
        })
        # max_history means conversation turns, where a turn is user+assistant.
        self._items = self._items[-(self.max_history * 2):]

    def as_messages(self):
        return list(self._items)

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)


def get_process_memory_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)
    except Exception:
        return None
