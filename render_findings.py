"""Render the findings and one figure from existing results; uses no remote compute."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rfm-findings-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rfm-findings-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
INK = "#202d39"
MUTED = "#556575"
BLUE = "#246d9f"


def excess_figure(results):
    pairs = ["jlens|nla", "nla|sae", "jlens|sae"]
    names = ["J-lens vs NLA", "NLA vs SAE", "J-lens vs SAE"]
    means = np.array([results["report"][p]["excess"] for p in pairs])
    intervals = np.array([results["prompt_cluster_bootstrap"][p]["excess_ci95"] for p in pairs])
    y = np.array([2, 1, 0])
    fig, ax = plt.subplots(figsize=(8.4, 3.25))
    fig.subplots_adjust(left=0.22, right=0.76, top=0.81, bottom=0.24)
    ax.axvline(0, color="#8797a5", lw=1, linestyle="--")
    ax.errorbar(means, y, xerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
                fmt="o", markersize=7, color=BLUE, ecolor=BLUE, elinewidth=2, capsize=4)
    ax.set_yticks(y, [f"{name}  (n={results['report'][p]['n']})" for name, p in zip(names, pairs)])
    ax.set_xlim(-0.01, 0.38)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(np.arange(0, 0.4, 0.1))
    ax.set_xlabel("Excess word overlap: matched − mismatched", fontsize=10, color=INK)
    ax.tick_params(axis="both", labelsize=10, colors=INK, length=0)
    ax.grid(axis="x", color="#e7edf2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#becbd5")
    for yi, p, mean, (lo, hi) in zip(y, pairs, means, intervals):
        ax.text(1.035, yi, f"+{mean:.3f}\n[{lo:.3f}, {hi:.3f}]", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10, color=INK, linespacing=1.5)
    fig.suptitle("Qwen3.6-27B: agreement above cross-prompt mismatches", fontsize=13,
                 fontweight="bold", color=INK, x=0.05, ha="left", y=0.98)
    fig.text(0.05, 0.035, "Points: observed excess. Lines: 95% prompt-cluster bootstrap intervals.\n"
             "12 prompts; labels held fixed. Annotation uncertainty is not included.", fontsize=9, color=MUTED)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"agreement_excess.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def clean(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip("*")


class Page:
    width, height = 595.276, 841.89
    left, right = 44, 44

    def __init__(self, number):
        self.fig = plt.figure(figsize=(self.width / 72, self.height / 72), facecolor="white")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()
        self.y = self.height - 44
        self.number = number
        self.fig.text(self.left / self.width, 22 / self.height,
                      "Qwen3.6-27B · exploratory agreement · 12 October 2026", fontsize=8, color=MUTED)
        self.fig.text((self.width - self.right) / self.width, 22 / self.height,
                      f"{number} / 3", fontsize=8, color=MUTED, ha="right")

    def text(self, text, size=10.1, weight="normal", color=INK, gap=7, indent=0):
        text = clean(text)
        font = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        available = (self.width - self.left - self.right - indent) * self.fig.dpi / 72
        lines = []
        line = ""
        for word in text.split():
            candidate = (line + " " + word).strip()
            measured = self.renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
            if line and measured > available:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        leading = size * 1.37
        for line in lines:
            self.fig.text((self.left + indent) / self.width, self.y / self.height,
                          line, fontsize=size, fontweight=weight, family="DejaVu Sans",
                          color=color, va="top")
            self.y -= leading
        self.y -= gap

    def heading(self, text, level):
        self.y -= 4 if level == 2 else 0
        self.text(text, size=16 if level == 1 else 12, weight="bold", gap=9)

    def table(self, results):
        height = 84
        ax = self.fig.add_axes([self.left / self.width, (self.y - height) / self.height,
                               (self.width - self.left - self.right) / self.width, height / self.height])
        ax.axis("off")
        rows = []
        for pair, name in [("jlens|nla", "J-lens vs NLA"), ("nla|sae", "NLA vs SAE"), ("jlens|sae", "J-lens vs SAE")]:
            r = results["report"][pair]
            rows.append([name, str(r["n"]), f"{r['matched']:.3f}", f"{r['mismatched']:.3f}",
                         f"+{r['excess']:.3f}", f"{r['z']:.2f}", f"{r['frac_above_own_null']:.0%}"])
        table = ax.table(cellText=rows, colLabels=["Comparison", "n", "Matched", "Mismatch", "Excess", "z", "Beats all"],
                         colWidths=[.265, .055, .14, .15, .135, .09, .165], cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8.6)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#dbe4eb")
            cell.set_linewidth(0.5)
            cell.set_text_props(color=INK)
            if row == 0:
                cell.set_facecolor("#eaf1f6")
                cell.set_text_props(weight="bold")
            elif row % 2:
                cell.set_facecolor("#f7fafc")
            if col == 0:
                cell.get_text().set_ha("left")
        self.y -= height + 10

    def chart(self, results):
        height = 155
        self.fig.text(self.left / self.width, self.y / self.height,
                      "Excess agreement with 95% prompt-cluster intervals", fontsize=9, weight="bold", color=INK, va="top")
        ax = self.fig.add_axes([(self.left + 126) / self.width, (self.y - 100) / self.height,
                               278 / self.width, 77 / self.height])
        pairs = ["jlens|nla", "nla|sae", "jlens|sae"]
        names = ["J-lens vs NLA", "NLA vs SAE", "J-lens vs SAE"]
        means = np.array([results["report"][p]["excess"] for p in pairs])
        bounds = np.array([results["prompt_cluster_bootstrap"][p]["excess_ci95"] for p in pairs])
        y = np.array([2, 1, 0])
        ax.axvline(0, lw=0.8, color="#8797a5", ls="--")
        ax.errorbar(means, y, xerr=np.vstack((means - bounds[:, 0], bounds[:, 1] - means)),
                    fmt="o", color=BLUE, markersize=4, lw=1.3, capsize=3)
        ax.set_yticks(y, [f"{name} (n={results['report'][p]['n']})" for p, name in zip(pairs, names)])
        ax.set_xlim(-0.01, 0.38)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.tick_params(length=0, labelsize=8, colors=INK)
        ax.set_xlabel("Matched − mismatched overlap", fontsize=8, color=INK, labelpad=3)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#becbd5")
        ax.grid(axis="x", color="#e7edf2", lw=0.7)
        ax.set_axisbelow(True)
        for yi, mean, (lo, hi) in zip(y, means, bounds):
            ax.text(1.025, yi, f"+{mean:.3f}\n[{lo:.3f}, {hi:.3f}]", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.8, color=INK)
        self.fig.text(self.left / self.width, (self.y - 135) / self.height,
                      "Labels held fixed; annotation uncertainty is not included. 12 prompt clusters; 4,000 replicates.",
                      fontsize=7.8, color=MUTED, va="top")
        self.y -= height + 5

    def finish(self, pdf):
        assert self.y > 40, f"Page {self.number} exceeds available height: {self.y:.1f}"
        print(f"Page {self.number}: bottom of content at {self.y:.1f} points")
        pdf.savefig(self.fig)
        plt.close(self.fig)


def render(markdown, results):
    first, rest = markdown.split("## Methods\n", 1)
    second, third = rest.split("## Prompts used — verbatim\n", 1)
    pages = [first, "## Methods\n" + second, "## Prompts used — verbatim\n" + third]
    with PdfPages(OUT / "research_findings.pdf", metadata={
        "Title": "Do three interpretability methods agree on Qwen3.6-27B?",
        "Subject": "Exploratory agreement findings; no causal steering",
        "Author": "Qwen agreement experiment",
    }) as pdf:
        for number, section in enumerate(pages, 1):
            page = Page(number)
            if number > 1:
                page.text("Qwen3.6-27B · research findings", size=9, color=MUTED, gap=12)
            blocks = re.split(r"\n\s*\n", section.strip())
            for block in blocks:
                if block.startswith("# "):
                    page.heading(block[2:], 1)
                elif block.startswith("## "):
                    page.heading(block[3:], 2)
                elif block.startswith("| Comparison"):
                    page.table(results)
                elif block.startswith("!["):
                    page.chart(results)
                elif re.match(r"\d+\. ", block):
                    for line in block.splitlines():
                        page.text(line, size=9.8, gap=3)
                elif block.startswith("*Figure"):
                    # Intervals and conditional uncertainty are already printed in the figure.
                    continue
                else:
                    page.text(block, size=9.5 if number == 3 else 10.1, gap=7)
            if number == 3:
                page.text("Source links: github.com/m9h/tri-lens; huggingface.co/ceselder/qwen3.6-27b-nla-rl; "
                          "huggingface.co/ceselder/qwen36-27b-sae2m-l42.", size=8, color=MUTED, gap=0)
            page.finish(pdf)


def main():
    results = json.loads((OUT / "sae_comparisons.json").read_text())
    assert not results["training_performed"] and not results["steering_performed"]
    excess_figure(results)
    markdown = (ROOT / "research_findings.md").read_text()
    render(markdown, results)
    print(f"Report: {OUT / 'research_findings.pdf'}")


if __name__ == "__main__":
    main()
