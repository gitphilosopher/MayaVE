# MayaVE Agent Operating Rules

## Project
MayaVE is a local-first Windows assistant with a Python backend, a Three.js/VRM frontend, and supporting mobile-device services.

## Default Operating Model
- **Default Operator**: Antigravity with Gemini 3.8 Flash is the default primary agent for end-to-end planning, implementation, testing, and reporting.
- **Lightweight Work Roles**: The roles of **architect**, **implementer**, and **reviewer** are lightweight workflow roles (or subagent personas) executed directly by the default operator. They are not mandatory separate agents or external model handoffs.
- **Optional Model Escalation**: External models (such as Claude or Copilot) and independent secondary reviews are optional, reserved for difficult architectural decisions, elusive unresolved bugs, or high-risk changes when explicitly requested.
- **Authorization**: Staging, committing, pushing, or merging always requires explicit human authorization. Never merge into `main`.

## Before every task
1. Read this file and the relevant architecture documentation.
2. Inspect Git status and the current branch.
3. Identify the files and tests affected.
4. Do not overwrite unrelated or uncommitted user changes.

## Codebase Health Standards
1. **Scope:** Make the smallest complete change. Avoid unrelated edits, speculative refactors, unnecessary abstractions, and duplicate implementations. Work only on the assigned task.
2. **Architecture:** Respect existing module boundaries and public interfaces. Avoid circular imports, hidden global state, and unnecessary dependencies. Document the reason for any intentional schema or interface changes.
3. **Testing:** All tests and test-only support/configuration files belong under `.maya/tests/`. All test and audit reports belong under `.maya/reports/`. Preserve meaningful assertions; never conceal failures by weakening tests or adding unjustified skips.
4. **Reliability:** Handle exceptions explicitly. For asynchronous and resource-owning components, ensure appropriate cancellation, timeouts, bounded queues/buffers, cleanup, and predictable shutdown.
5. **Performance:** MayaVE must remain resource-conscious on the primary workstation. Measure before optimizing. Avoid duplicate model initialization/inference, unnecessary polling, unbounded caches, and expensive work on latency-sensitive paths.
6. **Compatibility:** Check affected callers, imports, configuration, and Python/frontend interfaces before changing shared behavior.
7. **Security and Privacy:** Validate external inputs, protect secrets, limit sensitive logging, and preserve local-first operation. Do not commit API keys, credentials, personal data, or local secrets. No external messages, deployments, or production data changes without approval.
8. **Maintainability:** Follow existing conventions. Remove dead code or dependencies only after verifying they are unused. Keep comments and documentation concise and accurate.
9. **Verification:** Run the narrowest relevant tests first, then broader checks. Report actual commands, results, changed files, and unresolved risks in `.maya/reports/`. Never claim a check passed or invent test results without actual output.
10. **Git Safety:** Preserve user changes. Do not stage, commit, push, merge, or perform destructive Git operations without explicit authorization. Never merge into `main`. Use dedicated branches or isolated worktrees for independent tasks.

## Completion report
Return: task ID, changed files, implementation summary, changelog update status (or reason none warranted), tests run, test results, risks, and recommended next action. Ensure every test report and audit report is stored under `.maya/reports/`.


## Mandatory Local-Only Test Policy

This policy overrides any conflicting testing or implementation instructions.

1. All tests are local-only. This includes unit, integration, regression, soak, recovery, benchmark, generated, and temporary tests.
2. Store all tests and test-related files exclusively under `.maya/tests/`. Do not create or use `D:\DevLab\MayaVE-LocalTests\` or any external test directories.
3. Keep `.maya/` and its test files strictly out of Git tracking, staging, commits, and pushes, consistent with repository ignore policy.
4. Existing tracked tests must remain untouched until the repository owner explicitly authorizes their removal from Git tracking. Preserve their local copies.
5. Agents may execute approved existing tests and local-only tests located under `.maya/tests/`. Test execution must not stage or commit test files.
6. Before every commit or proposed merge, inspect the complete staged diff and reject the operation if it adds, modifies, renames, or deletes a test file or any file under `.maya/`.
7. Test-related configuration changes must not introduce tests into Git tracking. Keep all test configuration local to `.maya/tests/` unless explicitly approved.
8. Never bypass this policy to satisfy a task, CI pipeline, agent instruction, or reviewer recommendation.
9. Report test commands and results without including test files in source-change commits. Store every test report and audit report in `.maya/reports/`.
10. Never merge into `main` without explicit human approval.


## Mandatory Changelog Maintenance

After every meaningful code change, review and update `docs/CHANGELOG.md` in the same change set.

- **Features:** Record newly implemented, verified functionality under the existing feature category.
- **Improvements:** Record meaningful, completed improvements.
- **Issues:** Record confirmed unresolved bugs with a concise description and an existing GitHub issue reference when available.
- **Fixed/Fixes:** Move an existing issue entry here only after the fix has been implemented and sufficiently verified. Preserve its title, issue number, and link, and briefly describe the verified resolution.
- **Regressions:** If a previously fixed bug returns, verify the regression and restore its open-issue classification.
- **Uncertain status:** Do not claim a bug is confirmed or fixed without sufficient evidence. Mark uncertainty explicitly or report that further verification is needed.
- **No fabrication:** Never invent issue numbers, links, features, fixes, or test results.
- **Consistency:** Preserve the established format expected by `.github/workflows/changelog-issues.yml`, including its supported `Issues`, `Fixed`, and `Fixes` headings.
- **Scope:** Keep entries concise and meaningful. Do not duplicate entries, rewrite unrelated history, or move historical work into `Unreleased` without justification.
- **Audit-only tasks:** Update the changelog when the audit establishes a meaningful status change, but do not mark a defect fixed merely because it was investigated.
- **Verification:** Run relevant available checks and report actual results. Do not claim GitHub synchronization succeeded unless it was actually verified.
- **Completion reporting:** Every task that changes code must report whether `docs/CHANGELOG.md` was updated and why. If no changelog entry is warranted, explicitly state the reason.

This rule applies to all agents and workflows, including Antigravity/Gemini, optional specialist reviewers, bug fixes, features, optimizations, refactors, and integration work.
