#!/usr/bin/env python3
from pathlib import Path
import os, re, sys

TAGS = [t.strip() for t in os.getenv("COMPLETED_TAGS", "").split(",") if t.strip()]
if not TAGS:
    sys.exit(0)

roadmap = Path("ROADMAP.md")
pattern = re.compile(r"^- \[ \] <!--TASK:(?P<tag>[A-Z]+)-->")

updated = []
for line in roadmap.read_text().splitlines():
    m = pattern.match(line)
    if m and m.group("tag") in TAGS:
        line = line.replace("- [ ]", "- [x]", 1)
        if "~~" not in line:
            head, desc = line.split("-->", 1)
            line = f"{head}-->{'~~' + desc.strip() + '~~'}"
    updated.append(line)

roadmap.write_text("\n".join(updated))
