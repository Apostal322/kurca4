"""Check relative links, images and #anchors in all tracked Markdown files (GitHub slug rules)."""
import re
import subprocess
import sys
from pathlib import Path


def slug(h):
    h = re.sub(r"[^\w\- ]", "", h.strip().lower())  # GitHub: drop punctuation, keep word chars, '-', ' '
    return h.replace(" ", "-")


def anchors(md):
    return {slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", md.read_text(encoding="utf-8"), re.M)}


files = [Path(f) for f in subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True).stdout.split()]
files += [p for p in Path(".").glob("*.md") if p not in files] + [p for p in Path("docs").glob("*.md") if p not in files]
bad = 0
for f in sorted(set(files)):
    text = f.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # ignore code blocks
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        dest = (f.parent / path).resolve() if path else f.resolve()
        if not dest.exists():
            print(f"{f}: missing target {target}")
            bad += 1
        elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            print(f"{f}: missing anchor #{anchor} in {dest.name}")
            bad += 1
print(f"checked {len(set(files))} files, {bad} broken links")
sys.exit(1 if bad else 0)
