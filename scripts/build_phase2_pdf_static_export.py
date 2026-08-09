#!/usr/bin/env python3
"""Build a print-only Phase 2 export from the validated portable report."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from lxml import etree, html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "phase2"
CANONICAL_HTML = REPORT_DIR / "SRRD_Phase2_BlackBox_Report_2026-08-09.html"
ARTIFACT_JSON = REPORT_DIR / "artifact.json"
PRINT_HTML = REPORT_DIR / "SRRD_Phase2_BlackBox_Report_2026-08-09_print.html"
ASSET_DIR = REPORT_DIR / "pdf_assets"


def has_class(name: str) -> str:
    return (
        "contains(concat(' ', normalize-space(@class), ' '), "
        f"' {name} ')"
    )


def make_model_loss(rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("ood_nll")
    colors = ["#3568A8" if model != "SRRD-Bilevel" else "#C4613B" for model in frame["model"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=180)
    bars = ax.bar(frame["model"], frame["ood_nll"], color=colors, width=0.68)
    for bar, value in zip(bars, frame["ood_nll"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("OOD standardized Gaussian NLL (lower is better)")
    ax.set_ylim(0.49, 0.555)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.7, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_rotation(rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("angle_degrees")
    x = frame["angle_degrees"].to_numpy()
    y = frame["abs_psi_update"].to_numpy()
    low = frame["psi_ci_low"].to_numpy()
    high = frame["psi_ci_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=180)
    ax.fill_between(x, low, high, color="#3568A8", alpha=0.18, label="95% bootstrap CI")
    ax.plot(x, y, color="#3568A8", marker="o", linewidth=2.1, markersize=5.2, label="Mean |ψ_update|")
    ax.axhline(0.10, color="#C4613B", linestyle="--", linewidth=1.2, label="Zero-margin 0.10")
    ax.set_xlabel("Probe rotation (degrees)")
    ax.set_ylabel("Observable update interaction |ψ_update|")
    ax.set_xticks(x)
    ax.set_xlim(-2, 92)
    ax.set_ylim(0, 0.93)
    ax.grid(axis="both", color="#D8DDE3", linewidth=0.7, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def unwrap_source_values(root: etree._Element) -> None:
    nodes = root.xpath(
        f".//*[{has_class('portable-source-tooltip')} and {has_class('portable-source-value')}]"
    )
    for node in nodes:
        values = node.xpath(f".//*[{has_class('portable-source-value-text')}]")
        value = values[0].text_content() if values else node.text_content()
        tail = node.tail
        node.clear(keep_tail=True)
        node.text = value
        node.tail = tail


def separate_source_metadata(root: etree._Element) -> None:
    for node in root.xpath(f".//*[{has_class('portable-source-meta')}]"):
        node.text = f" — {(node.text or '').lstrip()}"


def normalize_pdf_symbols(root: etree._Element) -> None:
    """Use ASCII equivalents for symbols absent from the compact CJK font."""
    replacements = {
        "ψ": "psi",
        "ρ": "rho",
        "⇒": "implies",
        "≤": "<=",
        "≥": ">=",
        "→": "->",
        "⁶": "^6",
        "⁴": "^4",
    }
    for node in root.iter():
        if node.text:
            for old, new in replacements.items():
                node.text = node.text.replace(old, new)
        if node.tail:
            for old, new in replacements.items():
                node.tail = node.tail.replace(old, new)


def remove_nodes(root: etree._Element, xpath: str) -> None:
    for node in root.xpath(xpath):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def insert_chart(
    root: etree._Element,
    *,
    chart_id: str,
    image_src: str,
    subtitle: str,
    source_note: str,
) -> None:
    figures = root.xpath(f".//figure[@data-chart-id='{chart_id}']")
    if len(figures) != 1:
        raise RuntimeError(f"Expected one chart figure for {chart_id}")
    figure = figures[0]
    remove_nodes(figure, f".//*[{has_class('portable-table-scroll')}]")
    remove_nodes(figure, f".//*[{has_class('portable-table-note')}]")
    caption = figure.find("figcaption")
    if caption is not None:
        subtitle_node = etree.SubElement(caption, "span")
        subtitle_node.text = f" — {subtitle}"
    chart = etree.Element("div", {"class": "portable-static-chart"})
    etree.SubElement(
        chart,
        "img",
        {"src": image_src, "alt": subtitle, "class": "print-chart-image"},
    )
    note = etree.SubElement(chart, "p", {"class": "print-chart-source"})
    note.text = source_note
    figure.append(chart)


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(ARTIFACT_JSON.read_text(encoding="utf-8"))
    datasets = artifact["snapshot"]["datasets"]
    model_path = ASSET_DIR / "aligned_model_ood_loss.png"
    rotation_path = ASSET_DIR / "observation_rotation_curve.png"
    make_model_loss(datasets["chart_models"], model_path)
    make_rotation(datasets["rotation"], rotation_path)

    parsed = html.parse(str(CANONICAL_HTML))
    document = parsed.getroot()
    fallback_nodes = document.xpath("//*[@id='data-analytics-portable-fallback']")
    if len(fallback_nodes) != 1:
        raise RuntimeError("Canonical portable fallback was not found")
    fallback = deepcopy(fallback_nodes[0])
    unwrap_source_values(fallback)
    separate_source_metadata(fallback)
    normalize_pdf_symbols(fallback)
    remove_nodes(fallback, f".//*[{has_class('portable-inline-source')}]")
    remove_nodes(fallback, ".//details")
    insert_chart(
        fallback,
        chart_id="model-loss-chart",
        image_src="pdf_assets/aligned_model_ood_loss.png",
        subtitle="Mean over 400 paired seeds; lower loss is better.",
        source_note="Source: exact artifact snapshot derived from phase2_seed_metrics.csv.gz.",
    )
    insert_chart(
        fallback,
        chart_id="rotation-chart",
        image_src="pdf_assets/observation_rotation_curve.png",
        subtitle="Mean and 95% paired-bootstrap CI, 80 seeds per angle.",
        source_note="Source: exact artifact snapshot from phase2_rotation_summary.csv.",
    )

    source_headings = fallback.xpath(f".//*[{has_class('portable-sources')}]/h2")
    if source_headings:
        source_headings[0].text = "Sources and reproducibility"

    output_root = etree.Element("html", lang="ja")
    head = etree.SubElement(output_root, "head")
    etree.SubElement(head, "meta", charset="utf-8")
    title = etree.SubElement(head, "title")
    title.text = artifact["manifest"]["title"]
    for source_style in document.xpath("//style[@data-data-analytics-portable-fallback='true']"):
        head.append(deepcopy(source_style))
    style = etree.SubElement(head, "style")
    style.text = """
@page { size: A4; margin: 15mm 16mm 17mm; }
html, body { background: #fff !important; color: #1b232c !important; font-family: 'Noto Sans JP', 'DejaVu Sans', sans-serif; }
.portable-fallback { display: block !important; width: 100% !important; max-width: none !important; padding: 0 !important; }
.portable-page-header { position: static !important; width: 100% !important; height: auto !important; margin: 0 0 18px !important; padding: 0 0 13px !important; border-bottom: 1px solid #cbd2da !important; }
.portable-page-header h1 { overflow: visible !important; white-space: normal !important; font-size: 23px !important; line-height: 1.22 !important; }
.portable-description { display: block !important; margin-top: 6px !important; color: #59636e !important; }
.portable-surface-label { display: block !important; margin: 0 0 5px !important; color: #59636e !important; font-size: 10px !important; text-transform: uppercase; }
.portable-page-meta { display: block !important; margin-top: 6px !important; color: #59636e !important; font-size: 10px !important; }
.portable-block-stack { display: block !important; margin-top: 0 !important; }
.portable-block { margin: 0 0 17px !important; }
.portable-markdown { max-width: none !important; }
.portable-markdown h2 { margin: 20px 0 7px !important; font-size: 18px !important; break-after: avoid; }
.portable-markdown p, .portable-markdown li { font-size: 9.8pt !important; line-height: 1.46 !important; }
.portable-metric-grid { display: table !important; width: 100% !important; border-spacing: 6px !important; table-layout: fixed; }
.portable-metric-card { display: table-cell !important; width: 25% !important; padding: 10px !important; border: 1px solid #d5dbe2 !important; border-radius: 8px !important; vertical-align: top; }
.portable-metric-label { font-size: 8.5pt !important; }
.portable-metric-value { font-size: 17pt !important; }
.portable-card-description { display: block !important; margin-top: 5px !important; font-size: 7.5pt !important; color: #59636e !important; }
.portable-content-card { padding: 0 !important; }
.portable-visual-header strong, .portable-visual-header h2 { font-size: 13pt !important; }
.portable-visual-header span { display: block; margin-top: 3px !important; color: #59636e !important; font-size: 8.5pt !important; }
.portable-static-chart { margin: 0 !important; overflow: visible !important; }
.print-chart-image { display: block; width: 100%; max-width: 175mm; height: auto; margin: 0 auto; }
.print-chart-source { margin: 4px 0 0 !important; color: #59636e; font-size: 7.5pt; }
.portable-table-scroll { overflow: visible !important; border: 0 !important; }
.portable-table-scroll table { width: 100% !important; table-layout: fixed !important; }
.portable-table-scroll th, .portable-table-scroll td { padding: 4px 3px !important; white-space: normal !important; font-size: 7.2pt !important; overflow-wrap: anywhere; }
.portable-table-note { display: none !important; }
.portable-sources { display: block !important; margin-top: 20px !important; padding-top: 12px !important; border-top: 1px solid #cbd2da !important; }
.portable-sources h2 { font-size: 13pt !important; }
.portable-sources li, .portable-sources p, .portable-source-meta { font-size: 8pt !important; }
figure, .portable-metric-grid { break-inside: avoid; }
table { break-inside: auto; }
tr { break-inside: avoid; }
"""
    body = etree.SubElement(output_root, "body")
    body.append(fallback)

    PRINT_HTML.write_bytes(
        etree.tostring(
            output_root,
            encoding="utf-8",
            method="html",
            pretty_print=True,
            doctype="<!DOCTYPE html>",
        )
    )
    print(
        json.dumps(
            {
                "print_html": str(PRINT_HTML),
                "charts": [str(model_path), str(rotation_path)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
