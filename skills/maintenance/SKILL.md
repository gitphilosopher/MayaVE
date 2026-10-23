# MayaVE Autophage Maintenance Skill

## 1. Overview
The Autophage Maintenance Skill provides local-first, autonomous codebase inspection, planning, dependency tracking, verification, and integration capabilities for the MayaVE assistant workstation.

Maintenance is an autonomous subsystem skill packaged under `skills.maintenance`.

## 2. Boundaries & Directory Layout
To preserve workstation safety, Git hygiene, and policy invariants, the Autophage skill strictly separates implementation code from runtime state, tests, and reports:

- **Canonical Implementation**: `skills/maintenance/` (tracked in version control).
- **Runtime State & Ledgers**: `.maya/maintenance/` (`state.json`, `approvals.json`, `cache.json`, `active_session.json`, `pending_proposals/`, `repair_memory.jsonl`). Kept strictly untracked.
- **Local Tests**: `.maya/tests/` (all regression, lifecycle, unit, and E2E tests). Kept strictly untracked.
- **Audit & Verification Reports**: `.maya/reports/`. Kept strictly untracked.
- **Isolated Task Worktrees**: `.maya/worktrees/` (`.maya/worktrees/<task_id>`).
- **Temporary Repositories & Scratch Space**: `.maya/tmp/`.

## 3. Entry Points & CLI Usage
- **Canonical Module CLI**:
  ```powershell
  .venv\Scripts\python.exe -m skills.maintenance.coordinator [OPTIONS]
  ```
- **Canonical Script CLI**:
  ```powershell
  .venv\Scripts\python.exe skills/maintenance/coordinator.py [OPTIONS]
  ```
- **Compatibility Shim**:
  ```powershell
  .venv\Scripts\python.exe .maya/maintenance/coordinator.py [OPTIONS]
  ```

Common commands:
- Status check: `--status`
- Dry run: `--dry-run`
- Plan objective: `--plan-objective "<prompt>" [--local-model qwen2.5-coder:3b]`
- List proposals: `--list-plans`
- Approve plan: `--approve-plan <PLAN_ID> --authorize`
- Execute task: `--execute-task <TASK_ID> --authorize`
- Integrate task: `--integrate <TASK_ID> --authorize`
- Reconcile worktrees: `--reconcile`
- Cleanup preview: `--cleanup-preview` (or `--cleanup --dry-run`)
- Safe worktree cleanup: `--cleanup <TASK_ID> --authorize`
- Integration branches & pending merges: `--pending-merges`
- Session backups: `--list-backups`
- Restore state backup: `--restore-backup <BACKUP_PATH> --authorize`

## 4. Architectural Invariants
1. **Coordinator Authority**: `CoordinatorAPI` and `StateStore` are the sole authority for task state transitions.
2. **Durable Lifecycle Ledger**: Events are durably archived append-only to `.maya/maintenance/events.jsonl` with fsync persistence, deduplication by `event_id`/`idempotency_key`, and corrupt trailing-line tolerance.
3. **Rotating Session Backups**: Atomic session-start snapshots of `state.json` under `.maya/maintenance/backups/` preserve the last known valid state before mutation (bounded retention of 5 backups; refuses corrupt state; restore requires `--authorize`).
4. **Deterministic Plan Validation**: `TaskContractValidator` enforces schema v1.0.0, DAG acyclicity, risk ceilings (no HIGH/VERY_HIGH from planner), and forbids targeting maintenance internals (`skills/maintenance/`, `.maya/maintenance/`).
5. **Local-Only Test Policy**: All tests and test fixtures must strictly reside under `.maya/tests/`.
6. **Integration Branch Governance**: Commits are never made directly to `main`. Integration automatically redirects to dedicated task branches (`autophage/integrate-<task_id>`).
7. **Worktree Isolation & Safe Cleanup**: Work is conducted strictly inside isolated worktrees (`.maya/worktrees/<task_id>`). Cleanup requires 5 verified eligibility conditions, explicit operator authorization (`--authorize`), and non-force branch deletion (`git branch -d`).
8. **Crash Resilience & Recovery Chaos Handling**: Interrupted in-flight tasks (`IN_PROGRESS`, `VERIFYING`, `INTEGRATING`) and unverified integration commits move safely to `RECOVERY_REQUIRED` on restart; direct transitions from `RECOVERY_REQUIRED` to `IN_PROGRESS` are strictly forbidden without operator review.
