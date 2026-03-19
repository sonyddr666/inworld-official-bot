import copy
import json
import threading
from pathlib import Path
from typing import Any, Dict

from inworld_client import get_default_user_state


class LocalStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"users": {}})

    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"users": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"users": {}}

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_user_state(self, user_id: int) -> Dict[str, Any]:
        key = str(user_id)
        with self.lock:
            data = self._read()
            state = get_default_user_state()
            state.update(copy.deepcopy(data.get("users", {}).get(key, {})))
            return state

    def save_user_state(self, user_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        key = str(user_id)
        with self.lock:
            data = self._read()
            merged = get_default_user_state()
            merged.update(copy.deepcopy(state))
            data.setdefault("users", {})[key] = merged
            self._write(data)
            return merged

    def patch_user_state(self, user_id: int, **updates: Any) -> Dict[str, Any]:
        state = self.get_user_state(user_id)
        state.update(updates)
        return self.save_user_state(user_id, state)

    def get_global_config(self) -> Dict[str, Any]:
        with self.lock:
            data = self._read()
            return copy.deepcopy(data.get("global_config", {}))

    def save_global_config(self, config: Dict[str, Any]) -> None:
        with self.lock:
            data = self._read()
            data["global_config"] = copy.deepcopy(config)
            self._write(data)
