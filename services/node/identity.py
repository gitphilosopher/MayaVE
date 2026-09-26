"""
services/node/identity.py
Persists a stable device_id for this MayaVE install so MayaNode sees the
same identity across restarts (MayaNode's own api/sync.py docstring notes
device_id is "trusted as given" — no auth handshake exists yet, so a
stable, self-consistent id is what makes cursor/memory state coherent
across restarts today).

Not an authentication mechanism — see services/node/client.py's docstring
for the auth-ready header hook this is deliberately kept separate from.
"""
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _state_dir(configured: str | None) -> Path:
    return Path(configured) if configured else (Path.home() / "Maya" / "Node")


def resolve_device_id(state_dir: str | None = None) -> str:
    """
    Returns this install's persisted device_id, generating and saving one
    on first use. Best-effort: falls back to a fresh, unpersisted id
    (unique per call) if the directory can't be created/written rather
    than crashing startup over a filesystem hiccup — a session that never
    manages to persist an id just never has a stable identity with
    MayaNode until the underlying problem (permissions, disk) is fixed.
    """
    directory = _state_dir(state_dir)
    path = directory / "device_id.txt"
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        directory.mkdir(parents=True, exist_ok=True)
        new_id = f"mayave-{uuid.uuid4()}"
        path.write_text(new_id, encoding="utf-8")
        logger.info(f"Generated new MayaNode device_id: {new_id}")
        return new_id
    except OSError as e:
        fallback = f"mayave-ephemeral-{uuid.uuid4()}"
        logger.warning(
            f"Could not persist device_id ({e}) — using ephemeral id "
            f"{fallback} for this session only."
        )
        return fallback