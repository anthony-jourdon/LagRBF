#!/usr/bin/env python3
"""
Extract PETSc-style C docs blocks into Sphinx .rst pages.

Supported comment forms:
  /*@C ... @*/
  /*@  ... @*/

Usage:
  python [cdocs_parser.py](http://_vscodecontentref_/1) \
      --src src include \
      --out docs/source/api/c/generated \
      --index [index.rst](http://_vscodecontentref_/2)
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Match both /*@C ... @*/ and /*@ ... @*/
BLOCK_RE = re.compile(r"/\*@(?P<kind>[A-Za-z]?)(?P<body>.*?)@\*/", re.DOTALL)

# Function definition signature after doc block, up to opening brace
FUNC_PREFIX_RE = re.compile(
    r"""
    (?P<sig>
      [A-Za-z_][\w\s\*\n]*?
      \(
      [^;{}]*?
      \)
    )
    \s*\{
    """,
    re.VERBOSE | re.DOTALL,
)

TYPE_PREFIX_RE = re.compile(
    r"""
    (?P<decl>
      (?:typedef\s+)?(?:struct|enum)\s+[A-Za-z_][A-Za-z0-9_]*
    )
    \s*(?:\{|;)
    """,
    re.VERBOSE | re.DOTALL,
)

PARAM_LINE_RE = re.compile(
    r"^[+.\-]\s*([A-Za-z_]\w*)(?:\s*\(([^)]+)\))?\s*-\s*(.+)$"
)

#PARAM_LINE_RE = re.compile(r"^[+.\-]\s*([A-Za-z_]\w*)\s*-\s*(.+)$")

PETSC_LINKS = {
    "DM": "https://petsc.org/release/manualpages/DM/DM/",
    "DMSWARM": "https://petsc.org/release/manualpages/DMSWARM/DMSWARM/",
    "DMSwarmGetField": "https://petsc.org/release/manualpages/DMSWARM/DMSwarmGetField/",
    "DMSwarmRestoreField": "https://petsc.org/release/manualpages/DMSWARM/DMSwarmRestoreField/",
    "PetscErrorCode": "https://petsc.org/release/manualpages/Sys/PetscErrorCode/",
    "PetscInt": "https://petsc.org/release/manualpages/Sys/PetscInt/",
    "PetscReal": "https://petsc.org/release/manualpages/Sys/PetscReal/",
}

PETSC_TYPE_LINKS = {
    "DM": "https://petsc.org/release/manualpages/DM/DM/",
    "DMSWARM": "https://petsc.org/release/manualpages/DMSWARM/DMSWARM/",
    "LagRBFCtx": None,
    "Partition": None,
    "PartitionOfUnity": None,
    "AdvectionCtx": None,
    "CellInfo": None,
    "PetscErrorCode": "https://petsc.org/release/manualpages/Sys/PetscErrorCode/",
    "PetscInt": "https://petsc.org/release/manualpages/Sys/PetscInt/",
    "PetscReal": "https://petsc.org/release/manualpages/Sys/PetscReal/",
    "PetscBool": "https://petsc.org/release/manualpages/Sys/PetscBool/",
    "DMSwarmGetField": "https://petsc.org/release/manualpages/DMSWARM/DMSwarmGetField/",
    "DMSwarmRestoreField": "https://petsc.org/release/manualpages/DMSWARM/DMSwarmRestoreField/",
}

SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
C_TYPE_HINT_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")

@dataclass
class DocEntry:
    name: str
    summary: str
    kind: str = "function"  # function | type
    collective: str = ""
    input_params: list[tuple[str, str, str]] = field(default_factory=list)
    output_params: list[tuple[str, str, str]] = field(default_factory=list)
    level: str = ""
    notes: list[str] = field(default_factory=list)
    seealso: list[str] = field(default_factory=list)
    rst_blocks: list[str] = field(default_factory=list)
    signature: str = ""
    source_file: str = ""


def canonical_symbol(item: str) -> str:
    s = item.strip()
    if s.endswith("()"):
        s = s[:-2]
    return s


def classify_symbol(item: str) -> str:
    raw = item.strip()
    core = canonical_symbol(raw)

    # External PETSc links (try both raw and canonical forms)
    if raw in PETSC_TYPE_LINKS:
        return f"`{raw} <{PETSC_TYPE_LINKS[raw]}>`_"
    if core in PETSC_TYPE_LINKS:
        return f"`{core} <{PETSC_TYPE_LINKS[core]}>`_"

    # Function reference
    if raw.endswith("()") and SYMBOL_RE.match(core):
        return f":c:func:`{core}`"

    # Type reference heuristic (LagRBFCtx, PetscInt, DM, etc.)
    if SYMBOL_RE.match(core) and C_TYPE_HINT_RE.match(core):
        return f":c:type:`{core}`"

    return core


def format_type_name(type_name: str) -> str:
    clean = re.sub(r"\s+", " ", type_name.strip())
    if not clean:
        return ""

    clean = re.sub(r"\s*\*\s*", " *", clean)
    simple = clean.replace("const ", "").replace("volatile ", "")
    simple = simple.replace(" *", "").replace("*", "").strip()

    if simple in PETSC_TYPE_LINKS:
        url = PETSC_TYPE_LINKS[simple]
        if url:
            return f"`{simple} <{url}>`_"
        return f":c:type:`{simple}`"

    if simple in {"LagRBFCtx", "Partition", "PartitionOfUnity", "AdvectionCtx", "CellInfo"}:
        return f":c:type:`{simple}`"
    return f"``{clean}``"


def find_symbol_decl(source: str, start_pos: int) -> tuple[str, str] | None:
    tail = source[start_pos:]

    fm = FUNC_PREFIX_RE.search(tail)
    tm = TYPE_PREFIX_RE.search(tail)

    if fm and (not tm or fm.start() < tm.start()):
        return ("function", normalize_ws(fm.group("sig")))
    if tm:
        return ("type", normalize_ws(tm.group("decl")))
    return None


def type_display_name(type_sig: str) -> str:
    sig = type_sig.strip()
    m = re.match(r"typedef\s+.*\s+([A-Za-z_][A-Za-z0-9_]*)\s*;?$", sig)
    if m:
        return m.group(1)
    m = re.match(r"(?:struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", sig)
    if m:
        return m.group(1)
    return sig


def autolink_known_types(text: str, documented_types: set[str]) -> str:
    if not documented_types:
        return text

    def repl(match):
        word = match.group(0)
        if word in documented_types:
            return f":c:type:`{word}`"
        return word

    # Link standalone identifiers only
    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, text)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sanitize_lines(block_body: str) -> list[str]:
    # Preserve relative indentation for RST directives while removing common indent
    body = textwrap.dedent(block_body).strip("\n")
    lines = body.splitlines()

    # Trim leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Keep right-trim only; keep left indentation for rst directive bodies
    return [ln.rstrip() for ln in lines]


def parse_seealso_items(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def parse_block(block_body: str) -> DocEntry | None:
    lines = sanitize_lines(block_body)
    if not lines:
        return None

    first = lines[0].strip()
    if " - " not in first:
        return None

    name, summary = first.split(" - ", 1)
    entry = DocEntry(name=name.strip(), summary=summary.strip())

    section = "intro"
    for line in lines[1:]:
        low = line.strip().lower()

        # Keep empty lines only for RST passthrough
        if not line.strip():
            if section == "rst":
                entry.rst_blocks.append("")
            continue

        if low == "input parameters:":
            section = "input"
            continue
        if low in ("output parameter:", "output parameters:"):
            section = "output"
            continue
        if low.startswith("level:"):
            entry.level = line.split(":", 1)[1].strip()
            section = "other"
            continue
        if low == "notes:":
            section = "notes"
            continue
        if low == "rst:":
            section = "rst"
            continue
        if low.startswith(".seealso:"):
            refs = line.split(":", 1)[1].strip()
            entry.seealso.extend(parse_seealso_items(refs))
            section = "seealso"
            continue

        if low in ("collective", "not collective"):
            entry.collective = line.strip()
            continue

        if section == "input":
            m = PARAM_LINE_RE.match(line.strip())
            if m:
                entry.input_params.append((m.group(1), m.group(2) or "", m.group(3).strip()))
            continue

        if section == "output":
            m = PARAM_LINE_RE.match(line.strip())
            if m:
                entry.output_params.append((m.group(1), m.group(2) or "", m.group(3).strip()))
            continue

        if section == "notes":
            entry.notes.append(line.strip())
            continue

        if section == "seealso":
            # Allow multi-line seealso continuation
            entry.seealso.extend(parse_seealso_items(line.strip()))
            continue

        if section == "rst":
            # Keep as authored; Sphinx directives require preserved indentation
            entry.rst_blocks.append(line)

    # De-duplicate seealso while preserving order
    seen = set()
    uniq = []
    for item in entry.seealso:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    entry.seealso = uniq

    return entry


def find_signature(source: str, start_pos: int) -> str | None:
    tail = source[start_pos:]
    m = FUNC_PREFIX_RE.search(tail)
    if not m:
        return None
    return normalize_ws(m.group("sig"))


def rst_escape_title(s: str) -> str:
    return s.replace("\n", " ").strip()


def format_seealso_item(item: str) -> str:
    return f"- {classify_symbol(item)}"


def format_source_path(path_str: str, repo_root: Path) -> str:
    p = Path(path_str).resolve()
    try:
        return p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def entry_to_rst(entry: DocEntry, documented_types: set[str], repo_root: Path) -> str:
    title = rst_escape_title(entry.name)
    lines: list[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    if entry.kind == "function":
        lines.append(f".. c:function:: {entry.signature}")
    else:
        lines.append(f".. c:type:: {entry.name}")
    lines.append("")
    lines.append(f"   {autolink_known_types(entry.summary, documented_types)}")
    lines.append("")

    if entry.collective:
        lines.append(f"   **{entry.collective}**")
        lines.append("")

    if entry.input_params:
        lines.append("   **Input Parameters**")
        lines.append("")
        for name, ptype, desc in entry.input_params:
            if ptype:
                lines.append(f"   - ``{name}`` ({format_type_name(ptype)}) - {desc}")
            else:
                lines.append(f"   - ``{name}`` - {desc}")
        lines.append("")

    if entry.output_params:
        lines.append("   **Output Parameters**")
        lines.append("")
        for name, ptype, desc in entry.output_params:
            if ptype:
                lines.append(f"   - ``{name}`` ({format_type_name(ptype)}) - {desc}")
            else:
                lines.append(f"   - ``{name}`` - {desc}")
        lines.append("")

    if entry.level:
        lines.append("   **Level**")
        lines.append("")
        lines.append(f"   - {entry.level}")
        lines.append("")

    if entry.notes:
        lines.append("   **Notes**")
        lines.append("")
        for n in entry.notes:
            lines.append(f"   - {n}")
        lines.append("")

    if entry.rst_blocks:
        lines.append("   **Additional Documentation**")
        lines.append("")
        for raw in entry.rst_blocks:
            if raw:
                lines.append(f"   {raw}")
            else:
                lines.append("")
        lines.append("")

    if entry.seealso:
        lines.append("   **See also**")
        lines.append("")
        for s in entry.seealso:
            lines.append(f"   {format_seealso_item(s)}")
        lines.append("")

    lines.append("   **Source**")
    lines.append("")
    lines.append(f"   - ``{format_source_path(entry.source_file, repo_root)}``")
    lines.append("")

    return "\n".join(lines)


def find_c_files(src_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for d in src_dirs:
        files.extend(sorted(d.rglob("*.c")))
        files.extend(sorted(d.rglob("*.h")))

    seen = set()
    unique = []
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def extract_from_file(path: Path) -> list[DocEntry]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: list[DocEntry] = []

    for m in BLOCK_RE.finditer(text):
        body = m.group("body")
        entry = parse_block(body)
        if entry is None:
            continue

        decl = find_symbol_decl(text, m.end())
        if not decl:
            continue
        
        entry.kind, entry.signature = decl
        entry.source_file = str(path.as_posix())
        entries.append(entry)

    return entries


def write_outputs(entries: list[DocEntry], out_dir: Path, index_file: Path, repo_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    documented_types = {e.name for e in entries if e.kind == "type"}

    generated_files = []
    for e in entries:
        fname = f"{e.name}.rst"
        path = out_dir / fname
        path.write_text(entry_to_rst(e, documented_types, repo_root), encoding="utf-8")
        generated_files.append(fname)

    generated_files.sort()

    idx_lines = [
        "Generated C API",
        "===============",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for gf in generated_files:
        idx_lines.append(f"   {gf[:-4]}")
    idx_lines.append("")

    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text("\n".join(idx_lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="Source directories to scan")
    ap.add_argument("--out", required=True, help="Output directory for generated rst files")
    ap.add_argument("--index", required=True, help="Path to generated index.rst")
    args = ap.parse_args()

    src_dirs = [Path(s).resolve() for s in args.src]
    repo_root = Path(os.path.commonpath([str(p) for p in src_dirs]))
    out_dir = Path(args.out).resolve()
    index_file = Path(args.index).resolve()

    files = find_c_files(src_dirs)
    all_entries: list[DocEntry] = []
    for f in files:
        all_entries.extend(extract_from_file(f))

    all_entries.sort(key=lambda e: (e.name.lower(), e.source_file))
    write_outputs(all_entries, out_dir, index_file, repo_root)

    print(f"Generated {len(all_entries)} documented C entries in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())