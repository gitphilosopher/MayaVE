"""One-off: apply Batch 1 metadata edits to config/command_domains.json in place.
Usage (repo root):  python brain/router/patch_command_domains.py
Text-level edits so the file's formatting/seed order is untouched. Idempotent."""
from pathlib import Path
p = Path("config/command_domains.json")
s = p.read_text(encoding="utf-8")

def sub(old, new):
    global s
    if old in s and new not in s:
        assert s.count(old) == 1, f"ambiguous: {old!r}"
        s = s.replace(old, new)

# timer.create: duration is required (eval cases + test specs expect clarification)
sub('''"duration": {
              "type": "string",
              "required": false
            }''', '''"duration": {
              "type": "string",
              "required": true,
              "prompt": "How long should I set it for"
            }''')
# confirmation-gated operations (skills enforce the actual confirmation)
for op, legacy in (("shutdown", "shutdown"), ("restart", "restart")):
    sub(f'"{op}": {{\n          "legacy_intent": "{legacy}",',
        f'"{op}": {{\n          "legacy_intent": "{legacy}",\n          "requires_confirmation": true,')
sub('"delete": {\n          "legacy_intent": "note_delete",',
    '"delete": {\n          "legacy_intent": "note_delete",\n          "requires_confirmation": true,')
p.write_text(s, encoding="utf-8")
print("ok")
