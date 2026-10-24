"""
skills/maintenance/migration.py
Safe state migration from legacy AutoPhage 2.x stores to AutoPhage 3.0 MaintenanceLedger.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Optional
from skills.maintenance.ledger import MaintenanceLedger, TaskEntry, TaskStatus

logger = logging.getLogger(__name__)

class StateMigrator:
    """
    Handles migration of task states from legacy JSON stores to the 3.0 ledger.
    Supports migration from both campaign_state.json and state.json.
    """
    def __init__(self, repo_root: Path, ledger: MaintenanceLedger):
        self.repo_root = repo_root
        self.ledger = ledger
        self.maintenance_dir = repo_root / ".maya" / "maintenance"

    def migrate_legacy_state(self) -> int:
        """
        Orchestrates migration from all known legacy sources.
        Returns the number of tasks migrated.
        """
        total_migrated = 0

        # 1. Migrate from .maya/maintenance/state.json (Legacy Task Ledger)
        legacy_state_path = self.maintenance_dir / "state.json"
        if legacy_state_path.exists():
            logger.info(f"Migrating legacy state.json from {legacy_state_path}...")
            total_migrated += self._migrate_state_json(legacy_state_path)

        # 2. Migrate from .maya/maintenance/campaign_state.json (AutoPhage 2.0 Campaign Engine)
        campaign_state_path = self.maintenance_dir / "campaign_state.json"
        if campaign_state_path.exists():
            logger.info(f"Migrating campaign_state.json from {campaign_state_path}...")
            total_migrated += self._migrate_campaign_json(campaign_state_path)

        return total_migrated

    def _migrate_state_json(self, path: Path) -> int:
        """Migrates tasks from the flat state.json format."""
        migrated_count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tasks = data.get("tasks", {})

            current_ledger = self.ledger.load()

            for tid, tdata in tasks.items():
                if tid in current_ledger:
                    continue

                # Map legacy fields to TaskEntry
                # Legacy state.json tasks often have 'status', 'target_files', etc.
                self.ledger.update_task(
                    tid,
                    status=self._map_status(tdata.get("status")),
                    target_files=tdata.get("target_files", []),
                    attempts=tdata.get("attempts", 0),
                    worker=tdata.get("worker") or tdata.get("worker_tier")
                )
                migrated_count += 1

        except Exception as e:
            logger.error(f"Failed to migrate state.json: {e}")

        return migrated_count

    def _migrate_campaign_json(self, path: Path) -> int:
        """Migrates tasks from the CampaignCheckpoint format."""
        migrated_count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tasks = data.get("tasks", {})

            current_ledger = self.ledger.load()

            for tid, tdata in tasks.items():
                if tid in current_ledger:
                    continue

                # Map CampaignTaskState to TaskEntry
                self.ledger.update_task(
                    tid,
                    status=self._map_status(tdata.get("status")),
                    target_files=tdata.get("target_files", []),
                    attempts=tdata.get("attempts", 0),
                    worker=tdata.get("worker_tier")
                )
                migrated_count += 1

        except Exception as e:
            logger.error(f"Failed to migrate campaign_state.json: {e}")

        return migrated_count

    def _map_status(self, status: Any) -> TaskStatus:
        """Maps legacy status strings to AutoPhage 3.0 TaskStatus."""
        if not status:
            return TaskStatus.PENDING

        s = str(status).upper()
        mapping = {
            "PENDING": TaskStatus.PENDING,
            "IN_PROGRESS": TaskStatus.IN_PROGRESS,
            "VERIFIED": TaskStatus.VERIFIED,
            "READY_TO_INTEGRATE": TaskStatus.VERIFIED, # 3.0 consolidates these
            "FAILED": TaskStatus.FAILED,
            "NEEDS_APPROVAL": TaskStatus.NEEDS_APPROVAL,
            "BLOCKED": TaskStatus.NEEDS_APPROVAL,
            "PAUSED": TaskStatus.PENDING,
            "RECOVERY_REQUIRED": TaskStatus.RECOVERY_REQUIRED,
        }
        return mapping.get(s, TaskStatus.PENDING)
