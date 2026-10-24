"""
skills/maintenance/macro/cli.py
Command-Line Interface and Dispatch Engine for AutoPhage 2.0 (Phase 5).

Supported Commands:
- autophage scan [--subsystem <name>] [--output <path>] [--mechanical-only]
- autophage fix <findings.md> [--max-attempts <n>] [--authorize]
- autophage fix --issue <issue-id> [--backlog <findings.md>] [--authorize]
- autophage improve [--focus <area>] [--output <path>]
- autophage review [branch|diff]
- autophage status
- autophage resume
- autophage --help
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional
from rich.markup import escape

from .tui import MacroTUI, GRAY, MINT, AMBER, CORAL, PURPLE, to_ascii_safe
from .discover import run_discovery, DiscoveryResult
from .campaign import (
    CampaignConfig,
    CampaignEngine,
    CampaignStateManager,
    CampaignStatus,
)
from .agent import MacroAgent, MacroAgentConfig

# Optional repository imports
try:
    from skills.maintenance.reviewer_safeguards import ReviewerSafeguardScanner
    from skills.maintenance.reviewer import CopilotReviewerAdapter, ReviewResult
    from skills.maintenance.adapter import AntigravityAgentAdapter, resolve_antigravity_executable
    from skills.maintenance.models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus
except ImportError:
    try:
        from ..reviewer_safeguards import ReviewerSafeguardScanner
        from ..reviewer import CopilotReviewerAdapter, ReviewResult
        from ..adapter import AntigravityAgentAdapter, resolve_antigravity_executable
        from ..models import TaskDefinition, TaskPriority, TaskRisk, TaskStatus
    except ImportError:
        ReviewerSafeguardScanner = None
        CopilotReviewerAdapter = None
        AntigravityAgentAdapter = None
        resolve_antigravity_executable = None
        TaskDefinition = None
        TaskPriority = None
        TaskRisk = None
        TaskStatus = None


def resolve_repo_root(custom_root: Optional[Path | str] = None) -> Path:
    """Resolve repository root directory."""
    if custom_root:
        return Path(custom_root).resolve()
    # Default to 4 levels up from this file or current working directory
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / ".git").exists() or (candidate / "config").exists():
        return candidate
    return Path.cwd().resolve()


def check_model_availability(repo_root: Path) -> dict[str, str]:
    """
    Check availability of configured models and adapters:
    - Planner: qwen3.5:4b (Ollama)
    - Tier 1: qwen2.5-coder:3b (Ollama)
    - Tier 2: Antigravity / Google Flash 3.8 (agy.exe)
    - Reviewer: GitHub Copilot CLI (copilot.exe)
    Returns mapping of Role -> "Target | Status".
    """
    results: dict[str, str] = {}

    # Probe Ollama for Planner and Tier 1
    ollama_models: set[str] = set()
    ollama_reachable = False
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "AutoPhage-Probe/2.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                name = m.get("name", "")
                if name:
                    ollama_models.add(name)
                    if name.endswith(":latest"):
                        ollama_models.add(name[:-7])
            ollama_reachable = True
    except Exception:
        ollama_reachable = False

    # Planner: qwen3.5:4b
    planner_target = "qwen3.5:4b (Ollama)"
    if not ollama_reachable:
        results["Planner"] = f"{planner_target}|UNAVAILABLE"
    elif any("qwen3.5:4b" in m for m in ollama_models):
        results["Planner"] = f"{planner_target}|AVAILABLE"
    else:
        results["Planner"] = f"{planner_target}|UNAVAILABLE"

    # Tier 1: qwen2.5-coder:3b
    tier1_target = "qwen2.5-coder:3b (Ollama)"
    if not ollama_reachable:
        results["Tier 1 Implementation"] = f"{tier1_target}|UNAVAILABLE"
    elif any("qwen2.5-coder:3b" in m for m in ollama_models):
        results["Tier 1 Implementation"] = f"{tier1_target}|AVAILABLE"
    else:
        results["Tier 1 Implementation"] = f"{tier1_target}|UNAVAILABLE"

    # Tier 2: Antigravity / Flash 3.8
    tier2_target = "Antigravity CLI (agy.exe)"
    antigravity_bin = None
    if resolve_antigravity_executable:
        try:
            antigravity_bin = resolve_antigravity_executable()
        except Exception:
            antigravity_bin = None
    if not antigravity_bin:
        antigravity_bin = shutil.which("agy.exe") or shutil.which("agy")

    if antigravity_bin and (Path(antigravity_bin).is_file() or shutil.which(str(antigravity_bin))):
        results["Tier 2 Escalation"] = f"{tier2_target}|AVAILABLE"
    else:
        results["Tier 2 Escalation"] = f"{tier2_target}|UNAVAILABLE"

    # Reviewer: Copilot
    reviewer_target = "GitHub Copilot CLI (copilot.exe)"
    copilot_bin = shutil.which("copilot.exe") or shutil.which("copilot")
    if copilot_bin:
        results["Independent Reviewer"] = f"{reviewer_target}|AVAILABLE"
    else:
        results["Independent Reviewer"] = f"{reviewer_target}|UNAVAILABLE"

    return results


class AutoPhageCLI:
    """
    Main AutoPhage 2.0 command-line runner and controller.
    """

    def __init__(self, repo_root: Optional[Path | str] = None, tui: Optional[MacroTUI] = None) -> None:
        self.repo_root = resolve_repo_root(repo_root)
        self.tui = tui or MacroTUI()
        self.state_dir = self.repo_root / ".maya" / "maintenance"
        self.state_mgr = CampaignStateManager(self.state_dir)

    # --------------------------------------------------------------------------
    # Subcommands
    # --------------------------------------------------------------------------

    def cmd_scan(self, args: argparse.Namespace) -> int:
        """Execute repository discovery and output Markdown findings report."""
        subsystem = getattr(args, "subsystem", "all") or "all"
        mech_only = bool(getattr(args, "mechanical_only", False))
        out_dir = Path(args.output).resolve().parent if getattr(args, "output", None) else (self.repo_root / ".maya" / "maintenance")

        self.tui.console.print(f"[{PURPLE}]Starting AutoPhage 2.0 repository discovery...[/]")
        self.tui.console.print(f"[{GRAY}]Subsystem: {subsystem} | Static only: {mech_only}[/]\n")

        # Probe model availability if not mechanical-only
        tier1_agent = None
        if not mech_only:
            avail = check_model_availability(self.repo_root)
            if "AVAILABLE" in avail.get("Tier 1 Implementation", ""):
                tier1_agent = MacroAgent(config=MacroAgentConfig(model="qwen2.5-coder:3b"))
            else:
                self.tui.console.print(
                    f"[{AMBER}]Notice: Local model (qwen2.5-coder:3b) unavailable. Falling back to mechanical-only scan.[/]\n"
                )
                mech_only = True

        res: DiscoveryResult = run_discovery(
            repo_root=self.repo_root,
            subsystem=subsystem,
            mechanical_only=mech_only,
            output_dir=out_dir,
            agent=tier1_agent,
        )

        # If user specified exact output path, copy or rename to it
        report_file = res.report_path
        if getattr(args, "output", None) and res.report_path:
            target_path = Path(args.output).resolve()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path != res.report_path:
                shutil.copy2(res.report_path, target_path)
                report_file = target_path

        self.tui.print_findings(res.findings, report_path=report_file)

        if res.findings and report_file:
            self.tui.console.print(
                f"[{MINT}]To start a campaign on these findings, run:[/]\n"
                f"  [{PURPLE}]autophage fix {report_file}[/]\n"
                f"  [{PURPLE}]autophage fix --issue <issue_id> --backlog {report_file}[/]\n"
            )

        return 0

    def cmd_fix(self, args: argparse.Namespace) -> int:
        """Execute multi-issue campaign (Mode B) or single issue (Mode A)."""
        issue_id = getattr(args, "issue", None)
        backlog_path = getattr(args, "findings_file", None) or getattr(args, "backlog", None)
        authorized = bool(getattr(args, "authorize", False))
        tier1_only = bool(getattr(args, "tier1_only", False))
        max_attempts = int(getattr(args, "max_attempts", 3))

        config = CampaignConfig(
            max_attempts_per_task=max_attempts,
            is_tier2_authorized=authorized,
            require_authorization_for_tier2=True,
            tier1_only=tier1_only,
        )

        # Probe model availability and initialize workers
        avail = check_model_availability(self.repo_root)
        tier1_avail = "AVAILABLE" in avail.get("Tier 1 Implementation", "")
        tier2_avail = "AVAILABLE" in avail.get("Tier 2 Escalation", "")

        tier1_agent = None
        if tier1_avail:
            tier1_agent = MacroAgent(
                config=MacroAgentConfig(
                    model=config.tier1_model,
                    base_url=config.ollama_base_url,
                )
            )

        tier2_adapter = None
        if AntigravityAgentAdapter and tier2_avail:
            try:
                tier2_adapter = AntigravityAgentAdapter(authorized=authorized)
            except Exception:
                tier2_adapter = None

        def progress_cb(msg: str) -> None:
            self.tui.console.print(f"[{GRAY}]{escape(to_ascii_safe(msg))}[/]")

        engine = CampaignEngine(
            repo_root=self.repo_root,
            config=config,
            state_mgr=self.state_mgr,
            agent=tier1_agent,
            tier2_adapter=tier2_adapter,
            on_progress=progress_cb,
        )

        # Mode A: Single-issue execution
        if issue_id:
            resolved_backlog = Path(backlog_path).resolve() if backlog_path else (self.state_dir / "findings.md")
            if not resolved_backlog.exists():
                # Try finding in .maya/reports/
                reports = sorted((self.repo_root / ".maya" / "reports").glob("findings-*.md"))
                if reports:
                    resolved_backlog = reports[-1]

            if not resolved_backlog.exists():
                self.tui.console.print(
                    f"[{CORAL}]Error: Findings backlog file not found at '{resolved_backlog}'. "
                    f"Run 'autophage scan' first or pass --backlog <file>.[/]\n"
                )
                return 1

            self.tui.console.print(f"[{PURPLE}]AutoPhage Mode A: Executing single issue '{issue_id}'...[/]")
            self.tui.console.print(f"[{GRAY}]Backlog: {resolved_backlog}[/]\n")

            try:
                task_state = engine.execute_single_issue(issue_id, markdown_path=resolved_backlog)
            except Exception as exc:
                self.tui.console.print(f"[{CORAL}]Failed to execute issue '{issue_id}': {escape(to_ascii_safe(str(exc)))}[/]\n")
                return 1

            stat_val = getattr(getattr(task_state, "status", ""), "value", str(getattr(task_state, "status", "")))
            tier_val = getattr(getattr(task_state, "worker_tier", ""), "value", str(getattr(task_state, "worker_tier", "")))
            reason = task_state.blocking_reason or task_state.failure_reason or ""
            safe_reason = escape(to_ascii_safe(reason))

            self.tui.print_task_result(
                task_id=issue_id,
                status=stat_val,
                tier=tier_val,
                attempts=task_state.attempts,
                details=reason,
            )

            if task_state.status == TaskStatus.READY_TO_INTEGRATE:
                self.tui.console.print(
                    f"[{MINT}][OK] Issue {issue_id} verified in isolated worktree ({task_state.worktree_path}).[/]\n"
                    f"[{GRAY}]Mainline merge requires explicit human operator action per safety policy.[/]\n"
                )
                return 0
            elif task_state.status == TaskStatus.NEEDS_APPROVAL:
                self.tui.console.print(
                    f"[{AMBER}][!] Issue {issue_id} is awaiting human operator authorization: {safe_reason}[/]\n"
                    f"[{GRAY}]To authorize Tier 2 or held patch, run with --authorize.[/]\n"
                )
                return 2
            else:
                self.tui.console.print(
                    f"[{CORAL}][X] Issue {issue_id} execution failed: {safe_reason}[/]\n"
                )
                return 1

        # Mode B: Multi-issue campaign execution
        if not backlog_path:
            self.tui.console.print(
                f"[{CORAL}]Error: Please provide a findings markdown file path or specify --issue <id>.[/]\n"
                f"[{GRAY}]Example: autophage fix .maya/maintenance/findings.md[/]\n"
            )
            return 1

        resolved_file = Path(backlog_path).resolve()
        if not resolved_file.exists():
            self.tui.console.print(f"[{CORAL}]Error: Backlog file '{resolved_file}' does not exist.[/]\n")
            return 1

        self.tui.console.print(f"[{PURPLE}]AutoPhage Mode B: Initializing campaign from '{resolved_file.name}'...[/]")
        try:
            checkpoint = engine.ingest_markdown(resolved_file)
        except Exception as exc:
            self.tui.console.print(f"[{CORAL}]Failed to ingest backlog: {exc}[/]\n")
            return 1

        self.tui.console.print(
            f"[{MINT}]* Ingested {len(checkpoint.tasks)} tasks from {resolved_file.name}.[/]\n"
            f"[{GRAY}]Beginning single-concurrency campaign execution...[/]\n"
        )

        final_cp = engine.run_campaign()
        self.tui.print_status(final_cp)

        return 0 if final_cp.status == CampaignStatus.COMPLETED else 1

    def cmd_improve(self, args: argparse.Namespace) -> int:
        """Run read-only discovery of improvement opportunities."""
        focus = getattr(args, "focus", "all") or "all"
        out_file = getattr(args, "output", None) or (self.repo_root / ".maya" / "reports" / "improvements.md")
        out_file = Path(out_file).resolve()

        self.tui.console.print(f"[{PURPLE}]AutoPhage: Scanning for improvement opportunities (Focus: {focus})...[/]")
        self.tui.console.print(f"[{GRAY}]Read-only audit - code will not be mutated without authorization.[/]\n")

        avail = check_model_availability(self.repo_root)
        tier1_agent = None
        mech_only = False
        if "AVAILABLE" in avail.get("Tier 1 Implementation", ""):
            tier1_agent = MacroAgent(config=MacroAgentConfig(model="qwen2.5-coder:3b"))
        else:
            self.tui.console.print(
                f"[{AMBER}]Notice: Local model (qwen2.5-coder:3b) unavailable. Running mechanical discovery for improvements.[/]\n"
            )
            mech_only = True

        res = run_discovery(
            repo_root=self.repo_root,
            subsystem=focus,
            mechanical_only=mech_only,
            output_dir=out_file.parent,
            agent=tier1_agent,
        )

        if res.report_path and res.report_path != out_file:
            shutil.copy2(res.report_path, out_file)

        self.tui.print_findings(res.findings, report_path=out_file)

        self.tui.console.print(
            f"[{AMBER}]Notice: Improvement opportunities discovered. Implementation requires operator authorization.[/]\n"
            f"[{GRAY}]To implement a specific improvement safely in an isolated worktree, run:[/]\n"
            f"  [{PURPLE}]autophage fix --issue <id> --backlog {out_file} --authorize[/]\n"
        )
        return 0

    def cmd_review(self, args: argparse.Namespace) -> int:
        """Perform read-only review of a git diff, branch, or candidate patch."""
        target = getattr(args, "target", None)
        diff_text = ""

        if target:
            p = Path(target)
            if p.is_file():
                diff_text = p.read_text(encoding="utf-8", errors="replace")
            else:
                # Treat target as branch name
                try:
                    res = subprocess.run(
                        ["git", "diff", f"main...{target}"],
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    diff_text = res.stdout
                except Exception as exc:
                    self.tui.console.print(f"[{CORAL}]Failed to get diff for branch '{target}': {exc}[/]\n")
                    return 1
        else:
            # Current uncommitted changes
            try:
                res = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                diff_text = res.stdout
            except Exception as exc:
                self.tui.console.print(f"[{CORAL}]Failed to get git diff: {exc}[/]\n")
                return 1

        if not diff_text.strip():
            self.tui.console.print(f"[{MINT}][OK] No diff detected to review.[/]\n")
            return 0

        self.tui.console.print(f"[{PURPLE}]Conducting independent read-only review ({len(diff_text.splitlines())} diff lines)...[/]\n")

        tdef = TaskDefinition(
            task_id="MANUAL-REVIEW",
            title="Manual diff review",
            audit_source="cli",
            priority=TaskPriority.P2,
            risk=TaskRisk.LOW,
            description="CLI direct review",
            evidence="",
            target_files=[],
            test_targets=[],
        ) if TaskDefinition else None

        # Step 1: Safeguard Scanner
        safeguard_findings = []
        if ReviewerSafeguardScanner and tdef:
            scanner = ReviewerSafeguardScanner()
            if hasattr(scanner, "scan_diff"):
                scan_rep = scanner.scan_diff(tdef, diff_text)
            else:
                scan_rep = scanner.scan(tdef, diff_text)
            safeguard_findings = getattr(scan_rep, "findings", [])

        # Step 2: Copilot Reviewer if available
        recommendation = "APPROVE"
        summary = "Diff passed mechanical security review."
        findings = list(safeguard_findings)

        if CopilotReviewerAdapter and tdef:
            copilot = CopilotReviewerAdapter()
            if shutil.which(copilot.executable) or Path(copilot.executable).is_file():
                rev_res: ReviewResult = copilot.review(
                    task=tdef,
                    diff=diff_text,
                    verification_summary="CLI direct review",
                    worktree_path=self.repo_root,
                )
                recommendation = rev_res.recommendation.value
                summary = rev_res.summary
                findings.extend(rev_res.findings)
            else:
                summary += " (Copilot CLI not installed; performed AST & regex safeguard scan)."

        if safeguard_findings:
            recommendation = "NEEDS_HUMAN_APPROVAL"

        self.tui.print_review(recommendation, summary, findings)
        return 0

    def cmd_status(self, args: argparse.Namespace) -> int:
        """Display campaign and worker status."""
        cp = self.state_mgr.load()

        worktrees = []
        wt_dir = self.repo_root / ".maya" / "worktrees"
        if wt_dir.exists():
            worktrees = [d.name for d in wt_dir.iterdir() if d.is_dir()]

        model_statuses = check_model_availability(self.repo_root)
        self.tui.print_status(cp, worktrees=worktrees, model_statuses=model_statuses)
        return 0

    def cmd_resume(self, args: argparse.Namespace) -> int:
        """Safely resume an interrupted campaign."""
        cp = self.state_mgr.load()
        if not cp:
            self.tui.console.print(f"[{CORAL}]Error: No campaign state found to resume.[/]\n")
            return 1

        self.tui.console.print(f"[{PURPLE}]Resuming campaign '{cp.campaign_id}'...[/]\n")

        engine = CampaignEngine(repo_root=self.repo_root, state_mgr=self.state_mgr)
        try:
            resumed_cp = engine.resume_campaign()
            self.tui.print_status(resumed_cp)
            return 0 if resumed_cp.status == CampaignStatus.COMPLETED else 1
        except Exception as exc:
            self.tui.console.print(f"[{CORAL}]Failed to resume campaign: {exc}[/]\n")
            return 1


# ------------------------------------------------------------------------------
# Argument Parser Construction
# ------------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the AutoPhage 2.0 command parser."""
    parser = argparse.ArgumentParser(
        prog="autophage",
        description="AutoPhage 2.0 - MACRO Autonomous Maintenance Subsystem for MayaVE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autophage scan                             # Progressive repository discovery
  autophage scan --subsystem brain           # Audit a specific subsystem
  autophage fix backlog.md                   # Multi-issue campaign (Mode B)
  autophage fix --issue BUG-001              # Single-issue execution (Mode A)
  autophage fix --issue BUG-001 --authorize  # Authorize Tier 2 escalation
  autophage improve --focus skills           # Discover improvement candidates
  autophage review candidate.diff            # Read-only review of a diff
  autophage status                           # View active campaign & models
  autophage resume                           # Resume interrupted campaign
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="AutoPhage subcommand")

    # scan
    p_scan = subparsers.add_parser("scan", help="Run repository discovery and generate Markdown findings")
    p_scan.add_argument("--subsystem", default="all", help="Target subsystem to audit (default: all)")
    p_scan.add_argument("--output", default=None, help="Custom output path for findings Markdown report")
    p_scan.add_argument("--mechanical-only", action="store_true", help="Run mechanical static checks only")

    # fix
    p_fix = subparsers.add_parser("fix", help="Execute multi-issue campaign or single issue repair")
    p_fix.add_argument("findings_file", nargs="?", default=None, help="Markdown findings file to fix (Mode B)")
    p_fix.add_argument("--issue", default=None, help="Single issue ID to fix (Mode A)")
    p_fix.add_argument("--backlog", default=None, help="Backlog file containing the issue (Mode A)")
    p_fix.add_argument("--authorize", action="store_true", help="Authorize Tier 2 escalation / human approval gates")
    p_fix.add_argument("--tier1-only", action="store_true", help="Force execution with Tier 1 local model (qwen2.5-coder:3b) without cloud escalation")
    p_fix.add_argument("--max-attempts", type=int, default=3, help="Max repair attempts per task (default: 3)")

    # improve
    p_improve = subparsers.add_parser("improve", help="Discover and prioritize improvement opportunities")
    p_improve.add_argument("--focus", default="all", help="Subsystem focus area")
    p_improve.add_argument("--output", default=None, help="Custom output path for improvements backlog")

    # review
    p_review = subparsers.add_parser("review", help="Perform read-only review of a branch or diff")
    p_review.add_argument("target", nargs="?", default=None, help="Branch name, worktree, or diff file to review")

    # status
    subparsers.add_parser("status", help="Show campaign/task state, active work, and pending approvals")

    # resume
    subparsers.add_parser("resume", help="Safely resume an interrupted campaign")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for AutoPhage 2.0."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_cli_parser()

    # If no arguments provided, launch interactive REPL in TTY or show help
    if not argv:
        if sys.stdin.isatty():
            cli = AutoPhageCLI()
            return cli.tui.interactive_repl(dispatcher=lambda tokens: main(tokens))
        else:
            parser.print_help()
            return 0

    # Handle help commands directly
    if argv[0] in ("-h", "--help", "help"):
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    cli = AutoPhageCLI()

    if args.command == "scan":
        return cli.cmd_scan(args)
    elif args.command == "fix":
        return cli.cmd_fix(args)
    elif args.command == "improve":
        return cli.cmd_improve(args)
    elif args.command == "review":
        return cli.cmd_review(args)
    elif args.command == "status":
        return cli.cmd_status(args)
    elif args.command == "resume":
        return cli.cmd_resume(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
