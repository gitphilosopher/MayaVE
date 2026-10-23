"""
Atomic state persistence and process locking for MayaVE Maintenance Coordinator.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

try:
    from .ledger import LedgerValidationError, TaskLedger
    from .models import TaskDefinition
except ImportError:
    from ledger import LedgerValidationError, TaskLedger
    from models import TaskDefinition


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


class StateStore:
    """Manages atomic JSON file persistence and execution lock for TaskLedger."""

    def __init__(self, state_path: Path | str) -> None:
        self.state_path = Path(state_path).resolve()
        self.lock_path = self.state_path.with_suffix(".lock")
        self._thread_lock = threading.RLock()

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

    def save_ledger(self, ledger: TaskLedger) -> None:
        """
        Atomically save ledger to disk using temporary file replacement.
        Ensures valid graph before writing.
        """
        # 1. Validate in-memory graph integrity before writing
        ledger.validate_graph()
        payload = ledger.to_dict()

        # 2. Ensure parent directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Write to temporary sibling file
        temp_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        temp_file = self.state_path.with_name(f"{self.state_path.name}.tmp.{temp_id}")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # 4. Atomic file replacement with Windows transient lock backoff retry
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
