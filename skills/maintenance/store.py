"""
Atomic state persistence and process locking for MayaVE Maintenance Coordinator.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

try:
    from .ledger import LedgerValidationError, TaskLedger
    from .models import TaskDefinition
except ImportError:
    from ledger import LedgerValidationError, TaskLedger
    from models import TaskDefinition

logger = logging.getLogger("maya.maintenance.store")


class StateStoreError(Exception):
    """Base error for state persistence issues."""
    pass


class CorruptStateError(StateStoreError):
    """Raised when existing state file cannot be parsed or validated."""
    pass


class LockAcquisitionError(StateStoreError):
    """Raised when another coordinator process holds the active lock."""
    pass


def _is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is alive on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class EventLedgerArchive:
    """
    Append-only durable archive for authoritative lifecycle ledger events at .maya/maintenance/events.jsonl.
    Preserves all historical events beyond in-memory window, ensuring:
    - Ordered, append-only persistence.
    - Idempotency & deduplication: duplicate events (by event_id or idempotency_key) are never archived twice.
    - Fault tolerance: safely skips or handles partial lines, malformed JSON, and truncated records from past crashes.
    - Durability: write + flush + fsync; does not claim success unless the disk write completes.
    - Separation from TUI trace: strictly dedicated to authoritative lifecycle state transitions.
    """

    def __init__(self, archive_path: Path | str) -> None:
        self.archive_path = Path(archive_path).resolve()
        self._lock = threading.Lock()
        self._seen_event_ids: set[str] = set()
        self._seen_idempotency_keys: set[str] = set()
        self._initialized = False

    def _ensure_indexed(self) -> None:
        """Scan existing archive to index seen event IDs and keys, tolerating corrupt lines."""
        if self._initialized:
            return
        if not self.archive_path.exists():
            self._initialized = True
            return

        try:
            with open(self.archive_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        ev = json.loads(line_str)
                        if isinstance(ev, dict):
                            eid = ev.get("event_id")
                            if eid:
                                self._seen_event_ids.add(eid)
                            ikey = ev.get("idempotency_key")
                            if ikey:
                                self._seen_idempotency_keys.add(ikey)
                    except json.JSONDecodeError:
                        # Malformed or partially written line from past crash safely skipped
                        continue
        except Exception as e:
            logger.warning("Error reading event archive at %s: %s", self.archive_path, e)

        self._initialized = True

    def archive_events(self, events: list[dict]) -> int:
        """
        Durably append unarchived events to the events.jsonl archive.
        Returns the count of newly archived events.
        Raises StateStoreError if disk write fails.
        """
        if not events:
            return 0

        with self._lock:
            self._ensure_indexed()

            to_write = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                eid = ev.get("event_id")
                ikey = ev.get("idempotency_key")
                # Deduplication check
                if eid and eid in self._seen_event_ids:
                    continue
                if ikey and ikey in self._seen_idempotency_keys:
                    continue
                to_write.append(ev)

            if not to_write:
                return 0

            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Append with write + flush + fsync
                with open(self.archive_path, "a", encoding="utf-8") as f:
                    for ev in to_write:
                        line = json.dumps(ev, sort_keys=True)
                        f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                for ev in to_write:
                    eid = ev.get("event_id")
                    if eid:
                        self._seen_event_ids.add(eid)
                    ikey = ev.get("idempotency_key")
                    if ikey:
                        self._seen_idempotency_keys.add(ikey)

                return len(to_write)
            except Exception as e:
                raise StateStoreError(
                    f"Failed to durably append events to archive at '{self.archive_path}': {e}"
                ) from e

    def read_all_events(self) -> list[dict]:
        """Read all valid events in chronological order from the archive."""
        if not self.archive_path.exists():
            return []

        events = []
        with open(self.archive_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    ev = json.loads(line_str)
                    if isinstance(ev, dict):
                        events.append(ev)
                except json.JSONDecodeError:
                    continue
        return events

    def get_archived_count(self) -> int:
        """Return count of uniquely archived events."""
        with self._lock:
            self._ensure_indexed()
            return len(self._seen_event_ids)


class StateStore:
    """Manages atomic JSON file persistence, session backups, event archiving, and execution lock for TaskLedger."""

    def __init__(
        self,
        state_path: Path | str,
        archive_path: Path | str | None = None,
        backups_dir: Path | str | None = None,
    ) -> None:
        self.state_path = Path(state_path).resolve()
        self.lock_path = self.state_path.with_suffix(".lock")
        self._thread_lock = threading.RLock()
        archive_file = Path(archive_path).resolve() if archive_path else self.state_path.with_name("events.jsonl")
        self.archive = EventLedgerArchive(archive_file)
        self.backups_dir = Path(backups_dir).resolve() if backups_dir else (self.state_path.parent / "backups").resolve()

    def exists(self) -> bool:
        """Check if state file exists."""
        return self.state_path.is_file()

    def load_ledger(self) -> TaskLedger:
        """
        Load ledger from JSON file.
        Raises CorruptStateError if the file exists but cannot be safely parsed.
        """
        if not self.exists():
            raise FileNotFoundError(f"State file not found at: {self.state_path}")

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptStateError(
                f"State file at '{self.state_path}' is corrupted JSON: {e}"
            ) from e
        except Exception as e:
            raise CorruptStateError(
                f"Failed to read state file at '{self.state_path}': {e}"
            ) from e

        try:
            ledger = TaskLedger.from_dict(data)
            ledger.quarantine_legacy_unintegrated_verified()
            return ledger
        except LedgerValidationError as e:
            raise CorruptStateError(
                f"State file at '{self.state_path}' has invalid ledger schema: {e}"
            ) from e

    def save_ledger(self, ledger: TaskLedger, events: list[dict] | None = None) -> None:
        """
        Atomically save ledger to disk using temporary file replacement.
        Ensures valid graph before writing and durably archives all events to events.jsonl.
        """
        # 1. Validate in-memory graph integrity before writing
        ledger.validate_graph()

        # 2. Durably archive all events before writing state.json
        # Fails closed if event archive write fails
        all_events = list(ledger.events)
        if events:
            all_events.extend(events)
        self.archive.archive_events(all_events)

        payload = ledger.to_dict()

        # 3. Ensure parent directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Write to temporary sibling file
        temp_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        temp_file = self.state_path.with_name(f"{self.state_path.name}.tmp.{temp_id}")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # 5. Atomic file replacement with Windows transient lock backoff retry
            for attempt in range(5):
                try:
                    os.replace(temp_file, self.state_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise StateStoreError(f"Failed to atomically persist state to '{self.state_path}': {e}") from e

    def create_session_backup(self, max_backups: int = 5) -> Optional[Path]:
        """
        Create a rotating backup of state.json at session start.
        - Preserves the last known valid state before session mutation.
        - Uses atomic file write and bounded retention (keeps up to max_backups).
        - Validates that state.json is valid and non-empty; never overwrites or creates
          a backup from malformed/corrupted/empty state.
        - Never automatically restores an old backup over newer state.
        Returns the path of the created backup file, or None if no valid state exists to back up.
        """
        if not self.exists():
            return None

        # Check if empty (0 bytes)
        try:
            if self.state_path.stat().st_size == 0:
                logger.warning("State file at '%s' is empty; skipping backup creation.", self.state_path)
                return None
        except OSError:
            return None

        # Validate JSON content before backing up
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "tasks" not in data or "schema_version" not in data:
                logger.warning("State file at '%s' is missing required ledger schema; skipping backup.", self.state_path)
                return None
        except Exception as e:
            logger.warning(
                "State file at '%s' is corrupt (%s); refusing to overwrite backup set with corrupt state.",
                self.state_path,
                e,
            )
            return None

        # Prepare backups directory under .maya/maintenance/backups/
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"state_backup_{timestamp_str}.json"
        temp_file = self.backups_dir / f"{backup_name}.tmp.{uuid.uuid4().hex[:6]}"
        final_backup_path = self.backups_dir / backup_name

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, final_backup_path)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logger.error("Failed to write state backup: %s", e)
            return None

        # Bounded backup retention: prune older backups beyond max_backups
        try:
            existing_backups = sorted(
                [p for p in self.backups_dir.glob("state_backup_*.json") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_backup in existing_backups[max_backups:]:
                try:
                    old_backup.unlink()
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Failed to prune old backups: %s", e)

        return final_backup_path

    def list_backups(self) -> list[Path]:
        """Return existing valid backup paths sorted newest first."""
        if not self.backups_dir.is_dir():
            return []
        return sorted(
            [p for p in self.backups_dir.glob("state_backup_*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def restore_backup(self, backup_path: Path | str, authorized: bool = False) -> TaskLedger:
        """
        Restore state from an explicit backup file.
        Requires explicit operator authorization (--authorize).
        Validates backup integrity before atomic replacement.
        """
        if not authorized:
            raise PermissionError("Restoring a state backup requires explicit operator authorization (--authorize).")

        b_path = Path(backup_path).resolve()
        if not b_path.is_file():
            raise FileNotFoundError(f"Backup file not found at: {b_path}")

        with open(b_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ledger = TaskLedger.from_dict(data)

        # Atomically replace state.json
        self.save_ledger(ledger)
        return ledger

    def update_task_atomic(self, task: TaskDefinition) -> TaskLedger:
        """
        Thread/process-safe method to update a single task definition on disk
        by reloading the latest ledger, updating the task, and saving atomically.
        """
        with self._thread_lock:
            for attempt in range(5):
                try:
                    ledger = self.load_ledger()
                    persisted_task = ledger.get_task(task.task_id)
                    # Preserve this compatibility helper without allowing it to write status.
                    task = TaskDefinition.from_dict(task.to_dict())
                    task.status = persisted_task.status
                    ledger.update(task)
                    self.save_ledger(ledger)
                    return ledger
                except Exception as e:
                    if attempt == 4:
                        raise StateStoreError(
                            f"Failed to atomically update task '{task.task_id}': {e}"
                        ) from e
                    time.sleep(0.05 * (attempt + 1))
            return self.load_ledger()

    @contextlib.contextmanager
    def acquire_lock(self, force_unlock: bool = False) -> Iterator[None]:
        """
        Context manager to acquire an exclusive coordinator process lockfile.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_fd: int | None = None
        current_pid = os.getpid()
        lock_info = {
            "pid": current_pid,
            "acquired_at": str(os.getenv("COORDINATOR_LOCK_TIMESTAMP", "")),
        }

        # Check existing lockfile
        if self.lock_path.exists():
            stale = False
            existing_pid: int | None = None
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_pid = data.get("pid")
            except Exception:
                # Corrupt lockfile
                stale = True

            if existing_pid is not None and not _is_pid_alive(existing_pid):
                stale = True

            if stale or force_unlock:
                try:
                    self.lock_path.unlink()
                except OSError as e:
                    raise LockAcquisitionError(f"Could not remove stale lockfile: {e}") from e
            else:
                raise LockAcquisitionError(
                    f"Coordinator lock is active at '{self.lock_path}' held by PID {existing_pid}."
                )

        # Exclusive atomic creation
        try:
            lock_fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
            )
            with os.fdopen(lock_fd, "w", encoding="utf-8") as f:
                json.dump(lock_info, f)
                f.flush()
            lock_fd = None  # fdopen closed it
            yield
        except FileExistsError as e:
            raise LockAcquisitionError(
                f"Coordinator lock file '{self.lock_path}' already exists."
            ) from e
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            if self.lock_path.exists():
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass
