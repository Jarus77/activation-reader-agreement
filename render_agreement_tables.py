"""Render original and prompt-filtered agreement tables from saved measurements."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rfm-findings-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rfm-findings-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
PAIRS = [("jlens|nla", "J-lens vs NLA"), ("nla|sae", "NLA vs SAE"), ("jlens|sae", "J-lens vs SAE")]


def render(rows, filename, caption):
    fig = plt.figure(figsize=(15.2, 3.0), facecolor="#20242d")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    starts = [.026, .176, .245, .365, .490, .605, .690]
    centers = [.026, .192, .285, .418, .530, .625, .840]
    ends = [.153, .214, .335, .462, .575, .653, .978]
    headings = ["Pair", "n", "Matched", "Mismatched", "Excess", "z*", "Beats all mispairings"]
    for start, end, heading in zip(starts, ends, headings):
        ax.text(start, .845, heading, va="center", fontsize=15.5, fontweight="bold",
                family="DejaVu Sans Mono", color="#f4d89b")
        ax.plot([start-.008, end], [.747, .747], color="#505762", lw=1.2)
    for y, row in zip([.660, .489, .318], rows):
        for col, value in enumerate(row):
            ax.text(centers[col], y, value, ha="left" if col == 0 else "center", va="center",
                    fontsize=15, fontweight="bold", family="DejaVu Sans Mono", color="#e8e9ec")
        if y > .4:
            for start, end in zip(starts, ends):
                ax.plot([start-.008, end], [y-.088, y-.088], color="#454c58", lw=.9)
    ax.text(.026, .133, caption, color="#b9c2cf", fontsize=9.5, family="DejaVu Sans")
    ax.text(.026, .059, "* z is a descriptive effect size, not a significance-test statistic. SAE labels are automatic; results are exploratory.",
            color="#b9c2cf", fontsize=9.5, family="DejaVu Sans")
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{filename}.{suffix}", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    original = json.loads((OUT / "sae_comparisons.json").read_text())
    filtered = json.loads((OUT / "prompt_overlap.json").read_text())
    rows, new_rows = [], []
    for key, name in PAIRS:
        r=original["report"][key]
        s=filtered["report"][key]["scores"]["all_prompts_removed"]
        assert r["n"] == s["n"]
        rows.append([name, str(r['n']), f"{r['matched']:.3f}", f"{r['mismatched']:.3f}",
                     f"{r['excess']:+.3f}", f"{r['z']:+.2f}", f"{r['frac_above_own_null']:.0%}"])
        new_rows.append([name, str(s['n']), f"{s['matched']:.3f}", f"{s['mismatched']:.3f}",
                         f"{s['excess']:+.3f}", f"{s['z_descriptive']:+.2f}" if s['z_descriptive'] is not None else "n/a",
                         f"{s['fraction_beats_all_mismatches']:.0%}"])
    render(rows, "agreement_table_original", "Qwen3.6-27B · original overlap · same-activation pairs vs cross-prompt mismatches")
    render(new_rows, "agreement_table_prompt_removed", "Qwen3.6-27B · vocabulary of all 12 input prompts removed · overlap re-normalized; empty sets retained")
    print("| Pair | n | Matched | Mismatched | Excess | z* | Beats all mispairings |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in new_rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
