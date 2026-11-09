# -*- coding: utf-8 -*-
"""
Step 5 (combined) — Generate all publication figures (F1-F14).

This is a thin dispatcher that runs the three original step05 scripts in
order, then converts the resulting PNGs to SVG.

Original scripts (kept untouched for stability):
  step05_figures.py               F1, F2, F3, F4, F7, F8, F9, F10
  step05b_mgd_figures_nature.py   F5, F6
  step05_figures_nature.py        F11, F12, F13, F14 (JNER-only)

Output:
  results/figures/*.png    (300 dpi)
  results/figures/*.svg    (vector, editable in Illustrator)
"""
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = CODE_DIR.parent / "results" / "figures"
PAPER_DIRS = [
    CODE_DIR.parent / "paper" / "HumanBrainMapping",
    CODE_DIR.parent / "paper" / "JNER",
]

SCRIPTS = [
    ("step05_figures.py",               "F1 F2 F3 F4 F7 F8 F9 F10"),
    ("step05b_mgd_figures_nature.py",   "F5 F6"),
    ("step05_figures_nature.py",        "F1 F2 F3 F4 F7 F8 F9 F10"),
]


def run_one(script, only):
    print(f"\n{'=' * 70}")
    print(f"  Running {script}  (--only {only})")
    print(f"{'=' * 70}")
    items = only.split()
    any_failed = False
    for item in items:
        result = subprocess.run(
            [sys.executable, str(CODE_DIR / script), "--only", item],
            cwd=str(CODE_DIR),
        )
        if result.returncode != 0:
            any_failed = True
    return not any_failed


def png_to_svg():
    """Convert every PNG in results/figures/ to SVG via Inkscape if available,
    otherwise via pdf conversion. Pure-python fallback: nothing — SVG output
    is already produced by step05b and step05_figures_nature; this step only
    fills the gap for step05_figures.py which only writes PNG.
    """
    inkscape = None
    for cand in [r"C:\Program Files\Inkscape\bin\inkscape.exe",
                 r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe",
                 "/usr/bin/inkscape",
                 "/usr/local/bin/inkscape"]:
        if Path(cand).exists():
            inkscape = cand
            break

    if inkscape is None:
        print("[step05_all] Inkscape not found; SVG conversion skipped")
        print("  (F1-F4, F7-F10 will remain PNG-only — install Inkscape to convert)")
        return

    print(f"[step05_all] Converting PNG → SVG via {inkscape}")
    for png in sorted(FIGURE_DIR.glob("*.png")):
        svg = png.with_suffix(".svg")
        if svg.exists():
            continue
        subprocess.run([inkscape, str(png),
                        "--export-type=svg",
                        f"--export-filename={svg}"],
                       cwd=str(FIGURE_DIR),
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        print(f"  {png.name} → {svg.name}")


def main():
    ok = True
    for script, only in SCRIPTS:
        if not run_one(script, only):
            print(f"[step05_all] WARN: {script} reported errors (continuing)")
            ok = False

    print("\n[step05_all] All scripts done.")
    print("[step05_all] Converting PNGs to SVG…")
    png_to_svg()
    print("[step05_all] Done.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()