#!/usr/bin/env python3
"""Pre-submission sanity check for report.tex."""
import re
from pathlib import Path
from collections import Counter

ROOT      = Path(__file__).resolve().parent.parent
REPORT    = ROOT / "report" / "report.tex"
FIGURES   = ROOT / "report" / "figures"
BIB       = ROOT / "proposal" / "references.bib"
ACL_STY   = ROOT / "proposal" / "acl.sty"

tex = REPORT.read_text(encoding="utf-8")
bib = BIB.read_text(encoding="utf-8")
issues = []

# ── 1. Figures ────────────────────────────────────────────────────────────────
fig_refs = re.findall(r"includegraphics\[.*?\]\{(.*?)\}", tex)
print("=== FIGURES ===")
for r in fig_refs:
    p = ROOT / "report" / r
    ok = p.exists()
    if not ok:
        issues.append(f"Missing figure: {r}")
    print(f"  {'OK' if ok else 'MISSING'}: {r}")

# ── 2. Citations ──────────────────────────────────────────────────────────────
cite_raw = re.findall(r"citep?\{([^}]+)\}", tex)
cite_keys = set()
for k in cite_raw:
    for sub in k.split(","):
        cite_keys.add(sub.strip())
bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))
print("\n=== CITATIONS ===")
for k in sorted(cite_keys):
    ok = k in bib_keys
    if not ok:
        issues.append(f"Missing citation: {k}")
    print(f"  {'OK' if ok else 'MISSING'}: {k}")

# ── 3. Labels / refs ─────────────────────────────────────────────────────────
labels  = set(re.findall(r"\\label\{([^}]+)\}", tex))
ref_tgts = set(re.findall(r"\\ref\{([^}]+)\}", tex))
print("\n=== LABEL / REF ===")
for r in sorted(ref_tgts):
    ok = r in labels
    if not ok:
        issues.append(f"Undefined ref: {r}")
    print(f"  {'OK' if ok else 'UNDEF'}: {r}")

# ── 4. Triple hyphens ─────────────────────────────────────────────────────────
triple = len(re.findall("---", tex))
print(f"\n=== TRIPLE HYPHENS: {triple} (should be 0) ===")
if triple:
    issues.append(f"Triple hyphens: {triple}")
    for m in re.finditer("---", tex):
        ctx = tex[max(0, m.start()-30):m.end()+30].replace("\n", " ")
        print(f"  at: ...{ctx}...")

# ── 5. Begin / end balance ────────────────────────────────────────────────────
begins = re.findall(r"\\begin\{([^}]+)\}", tex)
ends   = re.findall(r"\\end\{([^}]+)\}", tex)
bc, ec = Counter(begins), Counter(ends)
unbalanced = [(e, bc[e], ec[e]) for e in sorted(set(bc) | set(ec)) if bc[e] != ec[e]]
print("\n=== BEGIN/END BALANCE ===")
if not unbalanced:
    print("  OK")
for e, b, en in unbalanced:
    issues.append(f"Unbalanced env \\{e}: begin={b}, end={en}")
    print(f"  UNBALANCED: {e}  begin={b}  end={en}")

# ── 6. Dependency files ───────────────────────────────────────────────────────
print("\n=== DEPENDENCIES ===")
for label, path in [("acl.sty", ACL_STY), ("references.bib", BIB)]:
    ok = path.exists()
    if not ok:
        issues.append(f"Missing dep: {label}")
    print(f"  {'OK' if ok else 'MISSING'}: {label}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== SUMMARY: {len(issues)} issue(s) ===")
for iss in issues:
    print(f"  - {iss}")
if not issues:
    print("  ALL CLEAR — ready for submission")
