"""
Structured Bug Backlog & Manifest Intake Engine for MayaVE Autophage.
Parses user-maintained Markdown defect backlogs (.md), strictly validates entries
against the canonical JSON manifest schema without hallucination or fabrication,
preserves provenance, and supports risk-gated queue admission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

try:
    from .hub import CoordinatorAPI
    from .models import (
        EvidenceRecord,
        PlanProposal,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
    )
    from .planner import compute_proposal_hash
    from .store import StateStore
    from .task_contract import (
        ALLOWED_SUBSYSTEM_ROOTS,
        FORBIDDEN_TARGET_EXACT,
        FORBIDDEN_TARGET_PREFIXES,
        TaskContractValidator,
    )
except ImportError:
    from hub import CoordinatorAPI
    from models import (
        EvidenceRecord,
        PlanProposal,
        TaskRisk,
        TaskStatus,
        utc_now_iso,
    )
    from planner import compute_proposal_hash
    from store import StateStore
    from task_contract import (
        ALLOWED_SUBSYSTEM_ROOTS,
        FORBIDDEN_TARGET_EXACT,
        FORBIDDEN_TARGET_PREFIXES,
        TaskContractValidator,
    )

_BUG_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_relpath(path_str: str) -> str:
    """Normalize a relative path string, stripping backticks and standardizing slashes."""
    p = path_str.strip().strip("`'\"")
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


class BacklogParseError(Exception):
    """Raised when a Markdown backlog violates formatting, schema, or safety invariants."""
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def parse_markdown_backlog(
    content: str,
    source_path: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """
    Parse a Markdown bug backlog into the canonical JSON manifest dictionary.
    Enforces deterministic extraction of bug identities, actual/expected behaviors,
    target files, test targets (.maya/tests/), priorities, and risks.

    Missing, ambiguous, or forbidden details are strictly FLAGGED as errors.
    No test targets, reproductions, or code changes are ever invented.
    """
    errors: list[str] = []
    lines = content.splitlines()

    # 1. Extract manifest ID and title from header if available
    manifest_id = "BACKLOG-INTAKE"
    manifest_desc = "Imported Markdown Bug Backlog"

    for line in lines[:15]:
        stripped = line.strip()
        if stripped.startswith("# "):
            header_text = stripped[2:].strip()
            # Check for pattern "# Backlog: <ID>" or similar
            if ":" in header_text:
                parts = header_text.split(":", 1)
                candidate_id = parts[1].strip()
                if _BUG_ID_REGEX.match(candidate_id):
                    manifest_id = candidate_id
                manifest_desc = header_text
            else:
                manifest_desc = header_text

    # 2. Split content into sections based on ## or ### defect headers
    # Matches patterns like "## BUG-001: Description" or "## [BUG-001] Description"
    header_pattern = re.compile(r"^(#{2,3})\s+(?:\[([A-Za-z0-9_-]+)\]|([A-Za-z0-9_-]+)\s*[:\-]\s*(.+)|(.+))$")

    defect_sections: list[tuple[str, str, list[str]]] = []
    current_id: Optional[str] = None
    current_title: Optional[str] = None
    current_lines: list[str] = []

    def _flush_section():
        nonlocal current_id, current_title, current_lines
        if current_id or current_title:
            defect_sections.append((current_id or "", current_title or "", list(current_lines)))
        current_id = None
        current_title = None
        current_lines = []

    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        if not in_code_block:
            m = header_pattern.match(line)
            if m:
                level = m.group(1)
                # Level 2 is primary defect separator; Level 3 only if current section is empty or starts a new defect
                if level == "##" or (level == "###" and (not current_id and not current_title)):
                    _flush_section()
                    if m.group(2):  # [BUG-001]
                        current_id = m.group(2).strip()
                        current_title = line.split("]", 1)[-1].strip(" :-")
                    elif m.group(3) and m.group(4):  # BUG-001: Title
                        current_id = m.group(3).strip()
                        current_title = m.group(4).strip()
                    else:  # Generic title
                        current_title = (m.group(5) or "").strip()
                    continue

        current_lines.append(line)

    _flush_section()

    parsed_bugs: list[dict[str, Any]] = []
    seen_bug_ids: set[str] = set()

    for sec_idx, (sec_id, sec_title, sec_body) in enumerate(defect_sections, start=1):
        if not sec_id and not sec_title and not "".join(sec_body).strip():
            continue

        # Parse key-value attributes and body sections
        bug_data: dict[str, Any] = {
            "bug_id": sec_id,
            "title": sec_title,
            "priority": "P2",
            "risk": "LOW",
            "target_files": [],
            "test_targets": [],
            "dependencies": [],
            "actual_behavior": "",
            "expected_behavior": "",
            "reproduction_steps": "",
            "acceptance_criteria": [],
        }

        # Sub-heading tracking
        current_subheading: Optional[str] = None
        subheading_text: dict[str, list[str]] = {
            "actual": [],
            "expected": [],
            "reproduction": [],
            "criteria": [],
        }

        in_sub_code = False
        active_list_attr: Optional[str] = None

        for raw_line in sec_body:
            s_line = raw_line.strip()
            if s_line.startswith("```"):
                in_sub_code = not in_sub_code
                if current_subheading:
                    subheading_text[current_subheading].append(raw_line)
                continue

            if not in_sub_code:
                # Check for sub-headings like "### Actual Behavior"
                if s_line.startswith("### "):
                    sub_title = s_line[4:].strip().lower()
                    active_list_attr = None
                    if "actual" in sub_title:
                        current_subheading = "actual"
                    elif "expected" in sub_title:
                        current_subheading = "expected"
                    elif "repro" in sub_title or "evidence" in sub_title:
                        current_subheading = "reproduction"
                    elif "criteria" in sub_title or "acceptance" in sub_title:
                        current_subheading = "criteria"
                    else:
                        current_subheading = None
                    continue

                # Check for inline key-value pairs (e.g. "**ID:** BUG-001" or "- **Priority:** P1")
                line_clean = s_line.lstrip("-* ").strip()
                if ":" in line_clean and not s_line.startswith("### "):
                    k_raw, v_raw = line_clean.split(":", 1)
                    k = k_raw.strip("*_` ").lower()
                    v = v_raw.strip("*_` ")

                    if k in ("id", "bug id", "bug_id", "task id"):
                        if not bug_data["bug_id"] and v:
                            bug_data["bug_id"] = v.strip("`'\"")
                        active_list_attr = None
                        continue
                    elif k in ("title", "summary"):
                        if not bug_data["title"] and v:
                            bug_data["title"] = v
                        active_list_attr = None
                        continue
                    elif k == "priority":
                        v_clean = v.strip("`'\"").upper()
                        if v_clean in ("P0", "P1", "P2", "P3"):
                            bug_data["priority"] = v_clean
                        else:
                            errors.append(f"Section #{sec_idx} ({sec_id or 'unknown'}): Invalid priority '{v}'. Must be P0, P1, P2, or P3.")
                        active_list_attr = None
                        continue
                    elif k == "risk":
                        v_clean = v.strip("`'\"").upper()
                        if v_clean in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH"):
                            bug_data["risk"] = v_clean
                        else:
                            errors.append(f"Section #{sec_idx} ({sec_id or 'unknown'}): Invalid risk '{v}'. Must be LOW, MEDIUM, HIGH, or VERY_HIGH.")
                        active_list_attr = None
                        continue
                    elif k in ("target files", "target_files", "targets"):
                        active_list_attr = "target_files"
                        if v:
                            for item in re.split(r"[,;]", v):
                                norm = normalize_relpath(item)
                                if norm and norm not in bug_data["target_files"]:
                                    bug_data["target_files"].append(norm)
                        continue
                    elif k in ("test targets", "test_targets", "tests"):
                        active_list_attr = "test_targets"
                        if v:
                            for item in re.split(r"[,;]", v):
                                norm = normalize_relpath(item)
                                if norm and norm not in bug_data["test_targets"]:
                                    bug_data["test_targets"].append(norm)
                        continue
                    elif k in ("dependencies", "deps"):
                        active_list_attr = "dependencies"
                        if v and v.lower() != "none":
                            for item in re.split(r"[,;]", v):
                                dep_id = item.strip("`'\" ")
                                if dep_id and dep_id not in bug_data["dependencies"]:
                                    bug_data["dependencies"].append(dep_id)
                        continue
                    elif k in ("actual behavior", "actual", "observed"):
                        current_subheading = "actual"
                        active_list_attr = None
                        if v:
                            subheading_text["actual"].append(v)
                        continue
                    elif k in ("expected behavior", "expected", "desired"):
                        current_subheading = "expected"
                        active_list_attr = None
                        if v:
                            subheading_text["expected"].append(v)
                        continue
                    elif k in ("reproduction", "reproduction steps", "evidence"):
                        current_subheading = "reproduction"
                        active_list_attr = None
                        if v:
                            subheading_text["reproduction"].append(v)
                        continue
                    elif k in ("acceptance criteria", "criteria"):
                        current_subheading = "criteria"
                        active_list_attr = "acceptance_criteria"
                        if v:
                            bug_data["acceptance_criteria"].append(v.strip())
                        continue

            # List item parsing for active list attribute
            if active_list_attr and (s_line.startswith("- ") or s_line.startswith("* ")):
                item_val = s_line[2:].strip()
                if active_list_attr in ("target_files", "test_targets"):
                    norm = normalize_relpath(item_val)
                    if norm and norm not in bug_data[active_list_attr]:
                        bug_data[active_list_attr].append(norm)
                elif active_list_attr == "dependencies":
                    dep_val = item_val.strip("`'\" ")
                    if dep_val and dep_val.lower() != "none" and dep_val not in bug_data["dependencies"]:
                        bug_data["dependencies"].append(dep_val)
                elif active_list_attr == "acceptance_criteria":
                    if item_val and item_val not in bug_data["acceptance_criteria"]:
                        bug_data["acceptance_criteria"].append(item_val)
                continue

            # Text accumulation under active subheading
            if current_subheading:
                if s_line.startswith("- ") or s_line.startswith("* "):
                    if current_subheading == "criteria":
                        crit = s_line[2:].strip()
                        if crit and crit not in bug_data["acceptance_criteria"]:
                            bug_data["acceptance_criteria"].append(crit)
                    else:
                        subheading_text[current_subheading].append(s_line)
                elif s_line:
                    subheading_text[current_subheading].append(s_line)

        # Assemble accumulated texts
        bug_data["actual_behavior"] = "\n".join(subheading_text["actual"]).strip()
        bug_data["expected_behavior"] = "\n".join(subheading_text["expected"]).strip()
        bug_data["reproduction_steps"] = "\n".join(subheading_text["reproduction"]).strip()

        # Strict Validation of Bug Entry — Do NOT fabricate missing details
        b_id = bug_data["bug_id"]
        if not b_id or not _BUG_ID_REGEX.match(b_id):
            errors.append(
                f"Section #{sec_idx}: Missing or invalid bug_id '{b_id}'. "
                "Each defect must declare a stable alphanumeric identifier (e.g., 'BUG-001')."
            )
            continue

        if b_id in seen_bug_ids:
            errors.append(f"Duplicate bug_id '{b_id}' found in backlog. Bug identifiers must be unique.")
            continue
        seen_bug_ids.add(b_id)

        title = bug_data["title"]
        if not title or len(title.strip()) < 5:
            errors.append(f"Bug '{b_id}': Missing or too short title (must be at least 5 chars).")

        if not bug_data["actual_behavior"]:
            errors.append(
                f"Bug '{b_id}': Missing 'Actual Behavior'. Backlog entries must describe the observed defect without fabrication."
            )

        if not bug_data["expected_behavior"]:
            errors.append(
                f"Bug '{b_id}': Missing 'Expected Behavior'. Backlog entries must describe the intended correct behavior."
            )

        if not bug_data["target_files"]:
            errors.append(
                f"Bug '{b_id}': Missing 'target_files'. At least one target file must be explicitly declared."
            )
        else:
            for tf in bug_data["target_files"]:
                if any(tf.startswith(p) for p in FORBIDDEN_TARGET_PREFIXES) or tf.lower() in FORBIDDEN_TARGET_EXACT:
                    errors.append(f"Bug '{b_id}': Target file '{tf}' modifies forbidden maintenance/system path.")
                elif not any(tf.startswith(sub) for sub in ALLOWED_SUBSYSTEM_ROOTS):
                    errors.append(f"Bug '{b_id}': Target file '{tf}' is outside allowed subsystems ({sorted(ALLOWED_SUBSYSTEM_ROOTS)}).")

        # Mandatory Local Test Policy validation
        if not bug_data["test_targets"]:
            errors.append(
                f"Bug '{b_id}': Missing 'test_targets'. Defect tasks require reproducible test evidence strictly under '.maya/tests/'."
            )
        else:
            for tt in bug_data["test_targets"]:
                if not tt.startswith(".maya/tests/"):
                    errors.append(
                        f"Bug '{b_id}': Test target '{tt}' violates local-only test policy. All tests must reside strictly under '.maya/tests/'."
                    )
                if ".." in tt:
                    errors.append(f"Bug '{b_id}': Test target '{tt}' contains forbidden path traversal.")

        # Default acceptance criteria if omitted
        if not bug_data["acceptance_criteria"]:
            bug_data["acceptance_criteria"] = [
                f"Fix confirmed defect '{title}' without regressions",
                "Target tests pass and maintain frozen test integrity",
            ]

        parsed_bugs.append(bug_data)

    if not parsed_bugs and not errors:
        errors.append("Markdown backlog contains zero defect entries.")

    manifest = {
        "manifest_id": manifest_id,
        "source_file": source_path,
        "description": manifest_desc,
        "bugs": parsed_bugs,
    }

    return manifest, errors


def manifest_to_contracts(
    manifest_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Convert canonical manifest bug entries into strict TaskContractValidator contracts.
    Preserves audit source provenance: f"manifest:{manifest_id}:{bug_id}".
    """
    manifest_id = manifest_data.get("manifest_id", "MANIFEST")
    bugs = manifest_data.get("bugs", [])
    contracts: list[dict[str, Any]] = []

    for b in bugs:
        bug_id = b["bug_id"]
        title = b["title"]
        actual = b.get("actual_behavior", "").strip()
        expected = b.get("expected_behavior", "").strip()
        repro = b.get("reproduction_steps", "").strip()

        desc_parts = [f"ACTUAL BEHAVIOR:\n{actual}", f"EXPECTED BEHAVIOR:\n{expected}"]
        description = "\n\n".join(desc_parts)

        evidence = repro if repro else actual

        contract = {
            "task_id": bug_id,
            "title": title[:140],
            "audit_source": f"manifest:{manifest_id}:{bug_id}",
            "priority": b.get("priority", "P2"),
            "risk": b.get("risk", "LOW"),
            "description": description,
            "evidence": evidence,
            "target_files": list(b.get("target_files", [])),
            "test_targets": list(b.get("test_targets", [])),
            "read_files": [],
            "dependencies": list(b.get("dependencies", [])),
            "acceptance_criteria": list(b.get("acceptance_criteria", [])),
            "verification_instructions": "Run pytest on test_targets; verify failing baseline then passing patch.",
            "task_type": "defect",
            "allow_file_deletion": False,
        }
        contracts.append(contract)

    # Validate batch integrity and DAG acyclicity
    is_valid, validation_errors = TaskContractValidator.validate_batch(contracts, is_planner=False)
    return contracts, validation_errors


def evaluate_risk_approval(
    risk: TaskRisk | str | None,
    risk_policy: str = "auto-low-medium",
) -> tuple[bool, str]:
    """
    Evaluates whether a task's authorization gate is automatically approved
    under the explicit risk-based approval policy.

    Rules:
    - LOW risk: automatically approve under policy.
    - MEDIUM risk: automatically approve under policy.
    - HIGH / VERY_HIGH risk: NEVER automatically approve; hold for explicit operator decision.
    - Missing, invalid, or uncertain risk: fails closed; requires human review.
    """
    if not risk:
        return False, "Missing risk classification; fails closed to human review."

    risk_str = risk.value.upper() if isinstance(risk, TaskRisk) else str(risk).upper()

    if risk_policy == "auto-low-medium":
        if risk_str in ("LOW", "MEDIUM"):
            return True, f"Automatically approved under explicit policy: 'auto-low-medium' (Risk: {risk_str})"
        elif risk_str in ("HIGH", "VERY_HIGH"):
            return False, f"{risk_str}-risk task requires explicit human decision under policy 'auto-low-medium'."
        else:
            return False, f"Unknown or uncertain risk '{risk_str}'; fails closed to human review."
    elif risk_policy == "manual":
        return False, "Manual operator authorization required (--authorize)."
    else:
        return False, f"Unknown risk approval policy '{risk_policy}'; fails closed."


def ingest_backlog_workflow(
    repo_root: Path,
    store: StateStore,
    backlog_path: Path,
    risk_policy: str = "auto-low-medium",
    authorized: bool = False,
    stage_only: bool = False,
    output_manifest_path: Optional[Path] = None,
) -> tuple[bool, dict[str, Any], str]:
    """
    Full intake workflow:
    1. Read Markdown backlog.
    2. Parse into canonical JSON manifest.
    3. Validate contracts and check duplicate task IDs against ledger.
    4. Write canonical JSON manifest to disk if requested.
    5. Generate PlanProposal with provenance hash.
    6. If stage_only or not authorized: save proposal for inspection.
    7. If authorized: apply risk-based approval policy to admit LOW/MEDIUM tasks to PENDING,
       and hold HIGH-risk tasks as NEEDS_APPROVAL.
    """
    if not backlog_path.is_file():
        return False, {}, f"Backlog file not found: {backlog_path}"

    try:
        content = backlog_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, {}, f"Failed to read backlog file: {e}"

    # 1. Parse Markdown
    manifest_data, parse_errs = parse_markdown_backlog(content, source_path=str(backlog_path.resolve()))
    if parse_errs:
        return False, {}, f"Backlog parsing failed with {len(parse_errs)} error(s):\n  - " + "\n  - ".join(parse_errs)

    # 2. Convert to contracts
    contracts, contract_errs = manifest_to_contracts(manifest_data)
    if contract_errs:
        return False, {}, "Task contract validation failed:\n  - " + "\n  - ".join(contract_errs)

    # 3. Deduplication check against existing ledger
    existing_ledger = store.load_ledger() if store.exists() else None
    existing_ids = set(existing_ledger.keys()) if existing_ledger else set()

    for c in contracts:
        tid = c["task_id"]
        if tid in existing_ids:
            return False, {}, f"Task ID collision: task '{tid}' already exists in ledger. Ingestion rejected to prevent duplicates."

    # 4. Save manifest JSON if requested or to default location
    if not output_manifest_path:
        manifest_dir = repo_root / ".maya" / "maintenance" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        output_manifest_path = manifest_dir / f"{manifest_data['manifest_id']}.json"

    try:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    except Exception as e:
        return False, {}, f"Failed to write canonical manifest JSON: {e}"

    # 5. Create PlanProposal
    plan_id = f"plan-{manifest_data['manifest_id'].lower()}"
    proposal_hash = compute_proposal_hash(manifest_data["description"], contracts)

    proposal = PlanProposal(
        plan_id=plan_id,
        objective=f"Manifest Intake [{manifest_data['manifest_id']}]: {manifest_data['description']}",
        tasks=contracts,
        status="PROPOSED",
        proposal_hash=proposal_hash,
        created_at=utc_now_iso(),
        max_tasks=len(contracts),
    )

    api = CoordinatorAPI(store)
    api.save_proposal(proposal)

    # 6. Evaluate authorization and risk-based admission
    if stage_only or not authorized:
        return True, {
            "plan_id": plan_id,
            "manifest_id": manifest_data["manifest_id"],
            "tasks_count": len(contracts),
            "manifest_json": str(output_manifest_path),
            "status": "PROPOSED",
            "registered_ids": [],
            "held_ids": [],
        }, (
            f"Backlog proposal '{plan_id}' staged successfully under pending_proposals/.\n"
            f"Inspect: python -m skills.maintenance.coordinator --inspect-plan {plan_id}\n"
            f"Approve: python -m skills.maintenance.coordinator --approve-plan {plan_id} --authorize"
        )

    # Authorized admission with risk-based approval policy
    if not store.exists():
        return False, {}, f"State store does not exist at: {store.state_path}"

    with store.acquire_lock():
        ledger = store.load_ledger()
        registered_ids: list[str] = []
        held_ids: list[str] = []

        for c_dict in contracts:
            t_def = TaskContractValidator.create_task_definition(c_dict)
            approved_by_policy, policy_reason = evaluate_risk_approval(t_def.risk, risk_policy=risk_policy)

            if approved_by_policy:
                t_def.status = TaskStatus.PENDING
                t_def.history.append(
                    EvidenceRecord(
                        timestamp=utc_now_iso(),
                        stage="APPROVAL_GATE",
                        details=policy_reason,
                        success=True,
                    )
                )
                registered_ids.append(t_def.task_id)
            else:
                t_def.status = TaskStatus.NEEDS_APPROVAL
                t_def.blocking_reason = policy_reason
                t_def.history.append(
                    EvidenceRecord(
                        timestamp=utc_now_iso(),
                        stage="APPROVAL_GATE",
                        details=f"Held in NEEDS_APPROVAL: {policy_reason}",
                        success=False,
                    )
                )
                held_ids.append(t_def.task_id)

            ledger.add_task(t_def)

        # Log coordinator ledger event
        idemp_reg = f"backlog-registered:{plan_id}"
        event_reg = api._event(
            task_id=f"plan:{plan_id}",
            event_type="PLAN_REGISTERED",
            previous="APPROVED",
            resulting="REGISTERED",
            actor=f"policy:{risk_policy}",
            outcome="accepted",
            idempotency_key=idemp_reg,
            details={
                "plan_id": plan_id,
                "manifest_id": manifest_data["manifest_id"],
                "registered_task_ids": registered_ids,
                "held_task_ids": held_ids,
                "risk_policy": risk_policy,
                "tasks_count": len(contracts),
            },
        )
        ledger.events = (getattr(ledger, "events", []) + [event_reg])[-5000:]
        store.save_ledger(ledger)

        # Update proposal state
        proposal.status = "REGISTERED"
        proposal.approved_at = utc_now_iso()
        proposal.approved_by = f"policy:{risk_policy}"
        proposal.registered_task_ids = registered_ids + held_ids
        api._write_proposal_atomic(proposal)

    return True, {
        "plan_id": plan_id,
        "manifest_id": manifest_data["manifest_id"],
        "tasks_count": len(contracts),
        "manifest_json": str(output_manifest_path),
        "status": "REGISTERED",
        "registered_ids": registered_ids,
        "held_ids": held_ids,
    }, (
        f"Manifest '{manifest_data['manifest_id']}' ingested into active ledger.\n"
        f"Admitted as PENDING (auto-approved): {registered_ids}\n"
        f"Held as NEEDS_APPROVAL: {held_ids}"
    )
