import json
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES = Path("config/candidates.jsonl")
REPORT = Path("logs/candidate_review_report.txt")


def load_candidates():
    rows = []

    with CANDIDATES.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append({
                    "_line": line_no,
                    "_error": "Invalid JSON",
                    "text": line,
                })
                continue

            row["_line"] = line_no
            rows.append(row)

    return rows


def main():
    if not CANDIDATES.exists():
        raise SystemExit(f"File not found: {CANDIDATES}")

    rows = load_candidates()

    valid = [r for r in rows if "_error" not in r]

    pending = [
        r for r in valid
        if r.get("verified") is not True
    ]

    verified = [
        r for r in valid
        if r.get("verified") is True
    ]

    intent_counts = Counter(
        r.get("intent", "<missing>")
        for r in valid
    )

    pending_by_intent = defaultdict(list)

    for row in pending:
        pending_by_intent[row.get("intent", "<missing>")].append(row)

    # Detect duplicate text, case-insensitive.
    text_groups = defaultdict(list)

    for row in valid:
        text = str(row.get("text", "")).strip().casefold()

        if text:
            text_groups[text].append(row)

    duplicates = {
        text: group
        for text, group in text_groups.items()
        if len(group) > 1
    }

    short_candidates = [
        r for r in pending
        if len(str(r.get("text", "")).strip()) < 8
    ]

    long_candidates = [
        r for r in pending
        if len(str(r.get("text", "")).strip()) > 200
    ]

    lines = []

    lines.append("MAYA VE11 — CANDIDATE REVIEW REPORT")
    lines.append("=" * 70)
    lines.append("READ-ONLY REPORT — candidates.jsonl was NOT modified.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total rows:       {len(rows)}")
    lines.append(f"Valid rows:       {len(valid)}")
    lines.append(f"Pending:          {len(pending)}")
    lines.append(f"Verified:         {len(verified)}")
    lines.append(f"Invalid JSON:     {len(rows) - len(valid)}")
    lines.append(f"Unique texts:     {len(text_groups)}")
    lines.append(f"Duplicate groups: {len(duplicates)}")
    lines.append("")

    lines.append("CANDIDATES BY INTENT")
    lines.append("-" * 70)

    for intent, count in sorted(
        intent_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):
        pending_count = len(pending_by_intent.get(intent, []))
        verified_count = count - pending_count

        lines.append(
            f"{intent:<30} "
            f"total={count:<5} "
            f"pending={pending_count:<5} "
            f"verified={verified_count}"
        )

    lines.append("")

    lines.append("PENDING CANDIDATES — GROUPED BY INTENT")
    lines.append("=" * 70)

    global_number = 0

    for intent in sorted(pending_by_intent):
        candidates = pending_by_intent[intent]

        lines.append("")
        lines.append(
            f"[{intent}] — {len(candidates)} pending"
        )
        lines.append("-" * 70)

        for row in candidates:
            global_number += 1

            text = str(row.get("text", "")).replace("\n", " ").strip()
            source = row.get("source", "")
            confidence = row.get("confidence", "")

            extra = []

            if source:
                extra.append(f"source={source}")

            if confidence != "":
                extra.append(f"confidence={confidence}")

            metadata = f" ({', '.join(extra)})" if extra else ""

            lines.append(
                f"{global_number:04d}. "
                f"line={row['_line']} "
                f"{text}{metadata}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("QUALITY FLAGS")
    lines.append("=" * 70)

    lines.append("")
    lines.append(
        f"Pending candidates shorter than 8 characters: "
        f"{len(short_candidates)}"
    )

    for row in short_candidates:
        text = str(row.get("text", "")).strip()
        lines.append(
            f"  line {row['_line']}: "
            f"[{row.get('intent', '<missing>')}] {text!r}"
        )

    lines.append("")
    lines.append(
        f"Pending candidates longer than 200 characters: "
        f"{len(long_candidates)}"
    )

    for row in long_candidates:
        text = str(row.get("text", "")).replace("\n", " ").strip()
        lines.append(
            f"  line {row['_line']}: "
            f"[{row.get('intent', '<missing>')}] {text}"
        )

    lines.append("")
    lines.append(
        f"Duplicate text groups: {len(duplicates)}"
    )

    for text, group in sorted(
        duplicates.items(),
        key=lambda x: (-len(x[1]), x[0])
    ):
        intents = ", ".join(
            sorted(set(str(r.get("intent", "<missing>")) for r in group))
        )

        lines.append("")
        lines.append(
            f"  DUPLICATE ({len(group)} rows) "
            f"intents=[{intents}]"
        )

        for row in group:
            lines.append(
                f"    line {row['_line']}: "
                f"{row.get('text', '')}"
            )

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"Report created: {REPORT.resolve()}")
    print(f"Total:    {len(valid)}")
    print(f"Pending:  {len(pending)}")
    print(f"Verified: {len(verified)}")
    print(f"Duplicate groups: {len(duplicates)}")


if __name__ == "__main__":
    main()