#!/usr/bin/env python3
"""Build the canonical report artifact for the SRRD Phase 2 experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase2_confirmatory"
OUTPUT = ROOT / "reports" / "phase2"


def summary_row(summary: pd.DataFrame, scenario: str, metric: str) -> pd.Series:
    rows = summary.loc[
        (summary["scenario"] == scenario) & (summary["metric"] == metric)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {scenario}/{metric}, got {len(rows)}")
    return rows.iloc[0]


def ci_text(row: pd.Series, digits: int = 4) -> str:
    return (
        f"{row['mean']:.{digits}f} "
        f"[{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "phase2_summary.csv")
    rotation = pd.read_csv(RESULTS / "phase2_rotation_summary.csv")
    seeds = pd.read_csv(RESULTS / "phase2_seed_metrics.csv.gz")
    metadata = json.loads((RESULTS / "phase2_metadata.json").read_text(encoding="utf-8"))
    gates = json.loads((RESULTS / "phase2_gates.json").read_text(encoding="utf-8"))
    budgets = json.loads((RESULTS / "phase2_model_budgets.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULTS / "phase2_validation.json").read_text(encoding="utf-8"))

    aligned = "true_srrd_aligned"
    r_ood = summary_row(summary, aligned, "r_ood")
    r_shuffle = summary_row(summary, aligned, "r_shuffle")
    r_frozen = summary_row(summary, aligned, "r_frozen")
    r_4x = summary_row(summary, aligned, "srrd_vs_flat_rnn_4x")
    psi = summary_row(summary, aligned, "abs_psi_update")
    smd = summary_row(summary, aligned, "state_max_smd")
    energy = summary_row(summary, aligned, "state_energy_distance")
    perm = summary_row(summary, aligned, "state_permutation_reject")

    headline = [
        {
            "r_ood": float(r_ood["mean"]),
            "r_shuffle": float(r_shuffle["mean"]),
            "r_frozen": float(r_frozen["mean"]),
            "abs_psi_update": float(psi["mean"]),
        }
    ]

    loss_columns = {
        "loss_markov_ssm": "Markov-SSM",
        "loss_flat_rnn_0_5x": "Flat RNN 0.5×",
        "loss_flat_rnn_1x": "Flat RNN 1×",
        "loss_flat_rnn_2x": "Flat RNN 2×",
        "loss_flat_rnn_4x": "Flat RNN 4×",
        "loss_adaptive_psr": "Adaptive PSR",
        "loss_history_mpc": "History-MPC",
        "loss_srrd_bilevel": "SRRD-Bilevel",
    }
    aligned_seeds = seeds.loc[seeds["scenario"] == aligned]
    budget_by_name = {item["name"]: item for item in budgets}
    budget_keys = {
        "Markov-SSM": "markov_ssm",
        "Flat RNN 0.5×": "flat_rnn_0_5x",
        "Flat RNN 1×": "flat_rnn_1x",
        "Flat RNN 2×": "flat_rnn_2x",
        "Flat RNN 4×": "flat_rnn_4x",
        "Adaptive PSR": "adaptive_psr",
        "History-MPC": "history_mpc",
        "SRRD-Bilevel": "srrd_bilevel",
    }
    model_rows = []
    for column, label in loss_columns.items():
        budget = budget_by_name[budget_keys[label]]
        loss = float(aligned_seeds[column].mean())
        model_rows.append(
            {
                "model": label,
                "family": budget["family"],
                "ood_nll": loss,
                "relative_to_best": loss / float(aligned_seeds["loss_strongest_baseline"].mean()),
                "feature_dim": int(budget["feature_dim"]),
                "trainable_parameters": int(budget["trainable_parameters"]),
                "rank": 0,
            }
        )
    model_rows.sort(key=lambda row: row["ood_nll"])
    for rank, row in enumerate(model_rows, start=1):
        row["rank"] = rank

    chart_models = [
        row for row in model_rows
        if row["model"] in {"Flat RNN 1×", "Flat RNN 2×", "Flat RNN 4×", "SRRD-Bilevel"}
    ]

    gate_specs = [
        ("G0", "Data integrity", "0 violations", "0"),
        ("G1", "State equivalence", "SMD/energy UCI ≤ .10; reject ≤ .10", f"{smd['ci_high']:.4f} / {energy['ci_high']:.4f} / {perm['mean']:.3f}"),
        ("G2", "Positive control", "aligned detected", f"|ψ| LCI {psi['ci_low']:.4f}"),
        ("G3", "Negative specificity", "designated nulls within margins", "all pass"),
        ("G4", "OOD superiority", "UCI(R_OOD) ≤ .90", f"{r_ood['ci_high']:.4f}"),
        ("G5", "Order necessity", "LCI(R_shuffle) ≥ 1.10", f"{r_shuffle['ci_low']:.4f}"),
        ("G6", "Update necessity", "LCI(R_frozen) ≥ 1.10", f"{r_frozen['ci_low']:.4f}"),
        ("G7", "Update interaction", "LCI(|ψ|) ≥ .20", f"{psi['ci_low']:.4f}"),
        ("G8", "Capacity robustness", "UCI(SRRD/RNN 4×) ≤ .90", f"{r_4x['ci_high']:.4f}"),
        ("G9", "Imbalance robustness", "confound separated", "SMD .4822; reject 1.00"),
        ("G10", "Observation boundary", "signature → 0", "ρ 1.00; 90° |ψ| .0328"),
    ]
    gate_rows = []
    for gate_id, label, criterion, evidence in gate_specs:
        key = next(key for key in gates["gates"] if key.startswith(gate_id + "_"))
        passed = bool(gates["gates"][key])
        gate_rows.append(
            {
                "gate": gate_id,
                "test": label,
                "criterion": criterion,
                "evidence": evidence,
                "status": "PASS" if passed else "FAIL",
            }
        )

    control_scenarios = [
        "true_srrd_aligned",
        "true_srrd_orthogonal",
        "frozen_rule",
        "order_invariant_memory",
        "flat_high_dim_markov",
        "persistent_history_no_update",
        "residual_state_imbalance",
        "pure_null",
    ]
    control_short_labels = {
        "true_srrd_aligned": "SRRD aligned",
        "true_srrd_orthogonal": "SRRD orthogonal",
        "frozen_rule": "Frozen-rule",
        "order_invariant_memory": "Order-invariant",
        "flat_high_dim_markov": "Flat Markov latent",
        "persistent_history_no_update": "Persistent history-only",
        "residual_state_imbalance": "Residual imbalance",
        "pure_null": "Pure null",
    }
    control_rows = []
    for scenario in control_scenarios:
        base = summary.loc[summary["scenario"] == scenario].iloc[0]
        control_rows.append(
            {
                "scenario": base["scenario_label"],
                "scenario_short": control_short_labels[scenario],
                "role": base["expected_role"],
                "n": int(base["n"]),
                "abs_psi_update": float(summary_row(summary, scenario, "abs_psi_update")["mean"]),
                "r_shuffle": float(summary_row(summary, scenario, "r_shuffle")["mean"]),
                "r_frozen": float(summary_row(summary, scenario, "r_frozen")["mean"]),
                "state_max_smd": float(summary_row(summary, scenario, "state_max_smd")["mean"]),
            }
        )

    rotation_rows = []
    for angle in sorted(rotation["angle_degrees"].unique()):
        subset = rotation.loc[rotation["angle_degrees"] == angle]
        values = {row["metric"]: row for row in subset.to_dict(orient="records")}
        rotation_rows.append(
            {
                "angle_degrees": int(angle),
                "coupling_cosine": float(subset["coupling_cosine"].iloc[0]),
                "abs_psi_update": float(values["abs_psi_update"]["mean"]),
                "psi_ci_low": float(values["abs_psi_update"]["ci_low"]),
                "psi_ci_high": float(values["abs_psi_update"]["ci_high"]),
                "abs_kappa_obs": abs(float(values["kappa_obs"]["mean"])),
                "r_shuffle": float(values["r_shuffle"]["mean"]),
                "srrd_vs_flat_rnn_4x": float(values["srrd_vs_flat_rnn_4x"]["mean"]),
                "n": int(values["abs_psi_update"]["n"]),
            }
        )

    generated_at = metadata["generated_at"]
    sources = [
        {
            "id": "phase2-summary",
            "label": "Phase 2 confirmatory bootstrap summary",
            "path": "results/phase2_confirmatory/phase2_summary.csv",
            "query": {
                "engine": "Python/pandas",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('results/phase2_confirmatory/phase2_summary.csv') ORDER BY scenario, metric;",
                "description": "Eight generator scenarios; paired seed-level estimates and 95% percentile-bootstrap intervals.",
                "executed_at": generated_at,
                "tables_used": ["phase2_summary.csv", "phase2_seed_metrics.csv.gz"],
                "filters": ["2,400 scenario-seed runs", "4,000 bootstrap resamples", "OOD C2=1.2 held out until final evaluation"],
                "metric_definitions": [
                    "R_OOD = SRRD OOD NLL / strongest baseline OOD NLL.",
                    "R_shuffle = shuffled-history SRRD NLL / intact-history SRRD NLL.",
                    "R_frozen = frozen-update SRRD NLL / intact-update SRRD NLL.",
                    "psi_update = History × randomized C1 probe interaction on the future response.",
                ],
            },
        },
        {
            "id": "phase2-rotation",
            "label": "Phase 2 observation-coupling sweep",
            "path": "results/phase2_confirmatory/phase2_rotation_summary.csv",
            "query": {
                "engine": "Python/pandas",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('results/phase2_confirmatory/phase2_rotation_summary.csv') ORDER BY angle_degrees, metric;",
                "description": "Seven preregistered observation rotations, 80 seeds per angle, with 95% bootstrap intervals.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "phase2-prereg",
            "label": "Frozen Phase 2 preregistration",
            "path": "preregistration/phase2.yaml",
            "query": {
                "description": "Claim, generators, model budgets, interventions, thresholds, and six-class decision rule frozen before the confirmatory run.",
                "executed_at": "2026-08-09T11:20:07Z",
            },
        },
        {
            "id": "phase2-gates",
            "label": "Phase 2 gate decisions",
            "path": "results/phase2_confirmatory/phase2_gates.json",
            "query": {
                "engine": "Python/json",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('results/phase2_confirmatory/phase2_gates.json');",
                "description": "Recorded outcomes for preregistered gates G0 through G10 and the six-class final decision.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "phase2-validator",
            "label": "Independent Phase 2 validation receipt",
            "path": "results/phase2_confirmatory/phase2_validation.json",
            "query": {
                "description": "Independent recomputation of hashes, ratios, model budgets, forbidden inputs, gates, and final classification.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "external-audit",
            "label": "Phase 2D external-dataset eligibility audit",
            "path": "external/phase2d_eligibility.json",
            "query": {
                "description": "Pre-outcome audit of Nano-drone, Bouc-Wen, and Silverbox against the confirmatory Phase 2D design requirements.",
                "executed_at": "2026-08-09",
            },
        },
        {
            "id": "phase1-report",
            "label": "SRRD Phase 1 falsification report",
            "path": "project_sources/02-SRRD_Phase1_Falsification_Report_2026-08-09-1-.pdf",
            "query": {
                "description": "User-provided Phase 1 report establishing SRRD/CVP separation and motivating the black-box test."
            },
        },
        {
            "id": "theory-audit",
            "label": "SRRD theory audit and reconstruction report",
            "path": "project_sources/01-SRRD-.pdf",
            "query": {
                "description": "User-provided audit defining state matching, rule-update interventions, and equivalent-capacity attacks."
            },
        },
    ]

    title = "SRRD Phase 2: Black-Box Operational Identifiability Falsification"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Latent-rule-blind confirmatory simulation with state matching, randomized probes, OOD intervention, ablations, and capacity attacks.",
            "generatedAt": generated_at,
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}\n\n**結果報告書 · 2026-08-09**"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": "phase2-summary",
                    "body": (
                        "## 技術要約\n\n"
                        "最終判定は **C — history dependence survives, SRRD decomposition does not** である。"
                        "true-SRRD/aligned generator では履歴順序と slow-update の必要性、History × probe の相互作用、"
                        "観測結合境界はいずれも検出された。しかし主評価の `R_OOD` は 1.0708 "
                        f"（95% CI {r_ood['ci_low']:.4f}–{r_ood['ci_high']:.4f}）で、事前規定した 0.90 以下に届かず、"
                        "SRRD-Bilevel は最強 baseline より約 7.1% 高損失だった。4× flat RNN にも負けたため、"
                        "履歴依存的な生成現象は支持されても、その予測に fast state + slow reconstructive state 分解が固有に必要だとは言えない。"
                    ),
                },
                {"id": "headline-strip", "type": "metric-strip", "cardIds": ["card-ood", "card-shuffle", "card-frozen", "card-update"]},
                {
                    "id": "decision",
                    "type": "markdown",
                    "sourceId": "phase2-prereg",
                    "body": (
                        "## 事前登録された中心命題は生存しなかった\n\n"
                        "中心命題は、観測状態を等価化した後でも、SRRD 型再構成が同等以上の容量を持つ flat latent-state、"
                        "predictive-state、history-control model では再現できない OOD 予測情報を与える、というものだった。"
                        "G4（OOD superiority）と G8（capacity robustness）が不通過であるため、"
                        "**operationally identified** とは判定しない。これは履歴効果そのものの否定ではなく、SRRD 分解の予測上の固有性の反証である。"
                    ),
                },
                {"id": "model-loss-chart-block", "type": "chart", "chartId": "model-loss-chart"},
                {"id": "model-table-block", "type": "table", "tableId": "model-table"},
                {
                    "id": "mechanism-separation",
                    "type": "markdown",
                    "sourceId": "phase2-summary",
                    "body": (
                        "## 機構感度は陽性、固有予測価値は陰性\n\n"
                        f"履歴順序を破壊すると損失は {r_shuffle['mean']:.2f} 倍、slow update を凍結すると {r_frozen['mean']:.2f} 倍になり、"
                        f"`|psi_update|` は {psi['mean']:.4f}（95% CI {psi['ci_low']:.4f}–{psi['ci_high']:.4f}）だった。"
                        "したがって被検 SRRD モデルが順序構造と更新項を実際に利用していたことは確認できる。"
                        "同時に、より単純な flat recurrent predictive state が OOD 予測を上回った。"
                        "この組合せが分類 C の核心である。"
                    ),
                },
                {"id": "control-table-block", "type": "table", "tableId": "control-table"},
                {
                    "id": "observation-boundary",
                    "type": "markdown",
                    "sourceId": "phase2-rotation",
                    "body": (
                        "## 観測結合境界はブラックボックス条件でも再現\n\n"
                        "観測 probe を 0° から 90° へ回転すると、結合 cosine の低下に伴って `|psi_update|` と `|kappa_obs|` は単調に消失した。"
                        "両者と coupling の Spearman ρ は 1.00。90° では `|psi_update|=0.0328`、`R_shuffle=1.0114` まで縮小した。"
                        "よって **SRRD exists ⇒ always observable** は Phase 2 でも成立しない。"
                    ),
                },
                {"id": "rotation-chart-block", "type": "chart", "chartId": "rotation-chart"},
                {
                    "id": "state-matching",
                    "type": "markdown",
                    "sourceId": "phase2-summary",
                    "body": (
                        "## State-matching gate\n\n"
                        f"aligned generator の最大 SMD は {smd['mean']:.4f}（UCI {smd['ci_high']:.4f}）、"
                        f"energy distance は {energy['mean']:.4f}（UCI {energy['ci_high']:.4f}）、"
                        f"paired permutation の棄却率は {perm['mean']:.3f} で、事前閾値 0.10 を満たした。"
                        "一方、residual-imbalance confound は SMD 0.4822、棄却率 1.00 と明確に検出された。"
                        "Gate 1 の棄却率は preregistration が point-rate 上限を規定していたため point rate で判定した。"
                        "初期 evaluator が bootstrap UCI を誤って適用した点は修正ログへ残し、閾値・データ・モデルを変えずに全実験を決定論的再実行した。"
                    ),
                },
                {"id": "gate-table-block", "type": "table", "tableId": "gate-table"},
                {
                    "id": "scope-method",
                    "type": "markdown",
                    "sourceId": "phase2-prereg",
                    "body": (
                        "## 範囲・データ・方法\n\n"
                        "評価モデルが受け取るのは `history, x_obs, c1, u2` のみで、true rule、latent rule、slow state、scenario mechanism は禁止した。"
                        "H1=A⁶B⁶C⁴ と H2=B⁶A⁶C⁴ は同じ回数と共通 suffix を持つ。C1 は probe/sham を無作為化し、"
                        "学習 C2 は {-0.8,-0.4,0.4,0.8}、OOD C2=1.2 は model selection・early stopping・tuning から完全 hold-out した。"
                        "8 generator × 計 2,400 scenario-seed runs、観測回転 560 runs、seed-level paired bootstrap 4,000 回である。"
                        "Primary loss は 6 horizon の standardized Gaussian NLL。nominal 1× models は 24 features / 156 trainable readout parameters に揃えた。"
                    ),
                },
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "phase2-validator",
                    "body": (
                        "## 独立検証\n\n"
                        "別 validator が preregistration SHA-256、seed ID 一意性、OOD contamination、"
                        "全 ratio、nominal/4× budget、禁止入力、11 gates、最終分類を再計算し、全検査に合格した。"
                        f"最大 ratio 再計算誤差は {max(validation['ratio_max_absolute_errors'].values()):.2e}。"
                        f"凍結 preregistration hash は `{metadata['preregistration_sha256']}` である。"
                    ),
                },
                {
                    "id": "external-data",
                    "type": "markdown",
                    "sourceId": "external-audit",
                    "body": (
                        "## 外部データ監査：Phase 2D は未実施\n\n"
                        "公開データを確認したが、履歴順序割付、観測 state matching 変数、randomized C1 probe/sham、"
                        "frozen OOD C2、post-intervention outcome を同時に満たす confirmatory dataset は見つからなかった。"
                        "[Nano-drone benchmark](https://arxiv.org/abs/2512.14450) は実世界の多入出力 OOD trajectory benchmark として二次評価に適するが、"
                        "history randomization と C1/sham がない。"
                        "[Bouc-Wen](https://www.nonlinearbenchmark.org/benchmarks/bouc-wen) は hysteresis を持つため history-without-reconstruction の負例候補、"
                        "[Silverbox](https://www.nonlinearbenchmark.org/benchmarks/silverbox) は非線形 system identification の二次 benchmark 候補である。"
                        "したがって外部データは参照・適格性監査に使い、Phase 2D 再現と誤表示しない。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 限界と反証範囲\n\n"
                        "本結果は synthetic latent-blind screening であり、実世界の SRRD 不在を示さない。"
                        "flat recurrent model は固定 reservoir core + 学習 readout で、完全学習 GRU/LSTM ではない。"
                        "Adaptive PSR と History-MPC も feature baseline であり、理論上の完全実装ではない。"
                        "それでも SRRD がこれらより優位でなかった事実は、現実装に有利な解釈を与えても中心命題を救わない。"
                        "また真の `r_t` はモデルへ隠したが generator 設計は人工的であり、latent parameter identifiability や unique ontology of self は検証対象外である。"
                    ),
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## 次の決定点\n\n"
                        "1. Phase 2B-hard として、end-to-end 学習 GRU/LSTM、learned PSR、実制御 History-MPC を nested tuning budget 付きで再実装する。\n"
                        "2. SRRD 側にも同じ optimization budget と calibration protocol を適用し、capacity ではなく分解の inductive bias を比較する。\n"
                        "3. Phase 2D は既存データへの後付け適用ではなく、history order と C1/sham を事前無作為化できる外部実験として設計する。\n"
                        "4. 現時点の科学的表現は『history matters; SRRD is not uniquely required』に固定する。"
                    ),
                },
            ],
            "cards": [
                {"id": "card-ood", "description": "SRRD NLL / strongest baseline NLL。0.90 以下が支持条件。", "dataset": "headline", "sourceId": "phase2-summary", "metrics": [{"label": "R_OOD", "field": "r_ood", "format": "number"}]},
                {"id": "card-shuffle", "description": "履歴順序を破壊したときの損失比。1.10 以上が必要。", "dataset": "headline", "sourceId": "phase2-summary", "metrics": [{"label": "R_shuffle", "field": "r_shuffle", "format": "number"}]},
                {"id": "card-frozen", "description": "slow update を凍結したときの損失比。1.10 以上が必要。", "dataset": "headline", "sourceId": "phase2-summary", "metrics": [{"label": "R_frozen", "field": "r_frozen", "format": "number"}]},
                {"id": "card-update", "description": "History × randomized probe interaction の絶対値。", "dataset": "headline", "sourceId": "phase2-summary", "metrics": [{"label": "|ψ_update|", "field": "abs_psi_update", "format": "number"}]},
            ],
            "charts": [
                {
                    "id": "model-loss-chart",
                    "title": "Aligned generator: competitive models' held-out OOD loss",
                    "subtitle": "Lower is better. Same 1× budget: 24 features / 156 trained readout parameters; 2× and 4× are capacity attacks.",
                    "type": "bar",
                    "dataset": "chart_models",
                    "sourceId": "phase2-summary",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Model"},
                        "y": {"field": "ood_nll", "type": "quantitative", "label": "OOD standardized NLL"},
                        "color": {"field": "model", "type": "nominal", "label": "Model"},
                        "tooltip": [
                            {"field": "relative_to_best", "type": "quantitative", "label": "Relative to strongest"},
                            {"field": "feature_dim", "type": "quantitative", "label": "Features"},
                            {"field": "trainable_parameters", "type": "quantitative", "label": "Trained parameters"},
                        ],
                    },
                    "layout": "full",
                    "maxRows": 20,
                },
                {
                    "id": "rotation-chart",
                    "title": "Observable update signature vanishes as probe coupling is rotated away",
                    "subtitle": "80 seeds per angle; line shows mean |ψ_update|. The source table retains bootstrap intervals and adjacent diagnostics.",
                    "type": "line",
                    "dataset": "rotation",
                    "sourceId": "phase2-rotation",
                    "encodings": {
                        "x": {"field": "angle_degrees", "type": "quantitative", "label": "Probe rotation (degrees)"},
                        "y": {"field": "abs_psi_update", "type": "quantitative", "label": "|ψ_update|"},
                        "tooltip": [
                            {"field": "coupling_cosine", "type": "quantitative", "label": "Observation coupling"},
                            {"field": "psi_ci_low", "type": "quantitative", "label": "95% CI low"},
                            {"field": "psi_ci_high", "type": "quantitative", "label": "95% CI high"},
                            {"field": "r_shuffle", "type": "quantitative", "label": "R_shuffle"},
                        ],
                    },
                    "layout": "full",
                    "maxRows": 20,
                },
            ],
            "tables": [
                {
                    "id": "model-table",
                    "title": "Full aligned-generator model competition",
                    "subtitle": "Mean OOD standardized NLL over 400 paired seeds; lower is better.",
                    "dataset": "models",
                    "sourceId": "phase2-summary",
                    "defaultSort": {"field": "rank", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "rank", "label": "Rank", "format": "number"},
                        {"field": "model", "label": "Model", "type": "text"},
                        {"field": "family", "label": "Family", "type": "text"},
                        {"field": "ood_nll", "label": "OOD NLL", "format": "number"},
                        {"field": "relative_to_best", "label": "vs best", "format": "number"},
                        {"field": "feature_dim", "label": "Features", "format": "number"},
                        {"field": "trainable_parameters", "label": "Trained params", "format": "number"},
                    ],
                },
                {
                    "id": "control-table",
                    "title": "Generator specificity and confound diagnostics",
                    "subtitle": "Means across 200 or 400 seeds. Aligned is the positive control; pure null, frozen, and order-invariant scenarios test specificity.",
                    "dataset": "controls",
                    "sourceId": "phase2-summary",
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "scenario_short", "label": "Generator", "type": "text"},
                        {"field": "abs_psi_update", "label": "|ψ|", "format": "number"},
                        {"field": "r_shuffle", "label": "R_shuffle", "format": "number"},
                        {"field": "r_frozen", "label": "R_frozen", "format": "number"},
                        {"field": "state_max_smd", "label": "Max SMD", "format": "number"},
                    ],
                },
                {
                    "id": "gate-table",
                    "title": "Preregistered Phase 2 gates",
                    "subtitle": "Nine gates passed; the two gates required for unique OOD value (G4 and G8) failed.",
                    "dataset": "gates",
                    "sourceId": "phase2-gates",
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "gate", "label": "Gate", "type": "text"},
                        {"field": "test", "label": "Test", "type": "text"},
                        {"field": "criterion", "label": "Criterion", "type": "text"},
                        {"field": "evidence", "label": "Observed", "type": "text"},
                        {"field": "status", "label": "Result", "type": "text"},
                    ],
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "models": model_rows,
                "chart_models": chart_models,
                "controls": control_rows,
                "rotation": rotation_rows,
                "gates": gate_rows,
            },
        },
    }

    target = OUTPUT / "artifact.json"
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
