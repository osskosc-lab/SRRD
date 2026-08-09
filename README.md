# SRRD Falsification Program — Phase 1 + Phase 2

SRRD を、構成妥当性だけでなくブラックボックス予測競争でも反証可能にするための合成実験プログラムです。

## Phase 2 — Black-Box Operational Identifiability Falsification

### 結論

事前登録した 8 generator・計 2,400 scenario-seed runs、観測回転 560 runs、4,000 回 paired bootstrap の最終判定は次です。

> **C — History dependence survives, SRRD decomposition does not.**

true-SRRD/aligned generator では履歴順序、slow update、History × probe 相互作用を検出しました。しかし SRRD-Bilevel は OOD 予測で最強 baseline を上回れず、4× flat RNN にも負けました。したがって、**履歴依存的な生成現象は残るが、fast state + slow reconstructive state という SRRD 分解が予測上固有に必要だとは支持されません。**

| 主指標 | 平均（95% bootstrap CI） | 事前条件 | 判定 |
|---|---:|---:|---|
| `R_OOD = L_SRRD / L_strongest` | 1.0708 [1.0670, 1.0746] | UCI ≤ 0.90 | **FAIL** |
| `R_shuffle` | 6.5665 [6.4966, 6.6365] | LCI ≥ 1.10 | PASS |
| `R_frozen` | 1.3155 [1.3097, 1.3213] | LCI ≥ 1.10 | PASS |
| `SRRD / Flat RNN 4×` | 1.0545 [1.0507, 1.0586] | UCI ≤ 0.90 | **FAIL** |
| `|psi_update|` | 0.8616 [0.8573, 0.8659] | LCI ≥ 0.20 | PASS |

aligned generator の平均 OOD standardized Gaussian NLL は、Flat RNN 2× が 0.5075、Flat RNN 4× が 0.5152、Flat RNN 1× が 0.5168、SRRD-Bilevel が 0.5433 でした。

### Gate 判定

| Gate | 内容 | 結果 |
|---|---|---|
| G0 | Data integrity | PASS |
| G1 | Observable-state equivalence | PASS |
| G2 | Positive-control detectability | PASS |
| G3 | Negative-control specificity | PASS |
| G4 | OOD superiority | **FAIL** |
| G5 | Order necessity | PASS |
| G6 | Reconstruction necessity | PASS |
| G7 | Update interaction | PASS |
| G8 | Capacity robustness | **FAIL** |
| G9 | Residual-imbalance robustness | PASS |
| G10 | Observation boundary | PASS |

観測 probe を 0° から 90° へ回転すると、`|psi_update|` は 0.8631 から 0.0328、`R_shuffle` は 6.4508 から 1.0114 まで消失しました。結合 cosine と `|psi_update|` / `|kappa_obs|` の Spearman rho はともに 1.00 で、Phase 1 の「SRRD が存在しても常に観測可能とは限らない」という境界を latent-rule-blind 条件でも再現しました。

### ブラックボックス契約

評価モデルへ渡すのは `history, x_obs, c1, u2` のみです。`true_rule, latent_rule, slow_state, scenario_mechanism` は禁止しました。

- 履歴: `H1=A^6 B^6 C^4`, `H2=B^6 A^6 C^4`
- randomized probe: `C1 ∈ {sham, probe}`
- 学習介入: `C2 ∈ {-0.8,-0.4,0.4,0.8}`
- 完全 hold-out OOD: `C2*=1.2`
- 現在状態: 近似 observable matching、exact reset 不使用
- 競合: Markov-SSM、flat recurrent capacity ladder、Adaptive PSR、History-MPC、SRRD-Bilevel

nominal 1× models は 24 features / 156 trained readout parameters、capacity attack は 2× と 4× に固定しました。

### 外部データ監査

公開 benchmark も確認しましたが、履歴順序割付、observable state matching、randomized C1 probe/sham、frozen OOD C2、post-intervention outcome を同時に満たす Phase 2D confirmatory dataset はありませんでした。

- [Nano-drone System Identification Benchmark](https://arxiv.org/abs/2512.14450): 実世界 OOD trajectory の二次予測 benchmark 候補
- [Bouc-Wen Hysteretic System](https://www.nonlinearbenchmark.org/benchmarks/bouc-wen): history-without-reconstruction の負例候補
- [Silverbox](https://www.nonlinearbenchmark.org/benchmarks/silverbox): 非線形 system identification の二次 benchmark 候補

したがって、外部データは適格性監査と次段階の候補選定に使い、**Phase 2D 再現とは表示しません。**

### Phase 2 の再現

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_phase2.py --mode confirmatory --output results/phase2_confirmatory
python scripts/verify_phase2.py --results results/phase2_confirmatory
```

主な出力:

- `preregistration/phase2.yaml`: 仮説、generator、介入、閾値、6分類判定
- `preregistration/phase2.freeze.json`: confirmatory run 前の SHA-256 freeze
- `results/phase2_confirmatory/phase2_seed_metrics.csv.gz`: 2,400 seed-level runs
- `results/phase2_confirmatory/phase2_summary.csv`: 平均と 95% CI
- `results/phase2_confirmatory/phase2_rotation_summary.csv`: observation-coupling sweep
- `results/phase2_confirmatory/phase2_gates.json`: G0–G10 と分類 C
- `results/phase2_confirmatory/phase2_validation.json`: 独立再計算 receipt
- `external/phase2d_eligibility.json`: 外部データ適格性監査
- `reports/phase2/SRRD_Phase2_BlackBox_Report_2026-08-09.html`: portable technical report
- `reports/phase2/SRRD_Phase2_BlackBox_Operational_Identifiability_Report_2026-08-09.pdf`: 5ページ結果報告書

### 反証範囲と限界

本実装の flat recurrent baseline は fixed reservoir core + trained readout であり、end-to-end 学習 GRU/LSTM ではありません。Adaptive PSR と History-MPC も feature baseline です。この Phase 2A–2C screening は実世界に SRRD が存在しないこと、真の latent parameter を同定できないこと、あるいは `unique ontology of self` を否定するものではありません。反証されたのは、**この事前登録実装で SRRD 分解が同等以上の flat predictive state に対して追加 OOD 予測価値を持つ**という中心命題です。

---

## Phase 1 — Construct Validity Falsification

SRRD → SRIV → CVP を「三つの競合理論」ではなく、**原理 → 力学的現象 → 観測量**として分離し、相互の同一視を反証するための最小合成実験です。

### 結論

400 seeds × 6 scenarios の事前登録型 Monte Carlo 実験では、全8 Gateが通過しました。

特に、意図的に構成した直交反例は次を同時に満たしました。

- 規則レベルの履歴分離: 2.6965（95% CI 2.6846–2.7087）
- 規則再構成率: 0.8339（95% CI 0.8330–0.8349）
- SRIV: 0.2728（95% CI 0.2711–0.2744）
- CVP: 0.0000（95% CI 0.0000–0.0000）
- 状態一致誤差: 0.0000

したがって、

\[
\boxed{\mathrm{SRRD}\Rightarrow\mathrm{CVP}}
\]

という**無条件含意は棄却**されます。生存するのは、観測写像・価値写像への非零結合を明示した条件付き階層です。

\[
\boxed{
\mathrm{SRRD}
\xrightarrow[\text{diagnostic coupling}\ne0]{ }
\mathrm{SRIV}
\xrightarrow[\text{viability coupling}\ne0]{ }
\mathrm{CVP}
}
\]

### 三層の独立定義

この Phase 1 では、略称の意味よりも測定対象を固定します。SRIV は *State-matched Rule-induced Intervention Variance*、CVP は *Counterfactual Viability Preservation* として操作化しています。

| 層 | 対象 | 入力として許されるもの | 禁止するもの |
|---|---|---|---|
| SRRD | 履歴依存の規則分離と、介入後の規則再構成 | 潜在規則目標と規則回復軌道 | 予測値、ラベル、効用、CVP |
| SRIV | 状態一致後に生じる介入応答分布の差 | 診断プローブ上の対予測分布 | ラベル、効用、潜在規則 |
| CVP | 再構成あり／なしの反実仮想的な生存機能差 | 独立した評価データのラベルと予測値 | 履歴符号、潜在規則、SRRD、SRIV |

実装上も各量は別関数で計算し、引数を共有しません。

### 最小数理モデル

#### 1. 履歴順序の符号化

同じ要素数を持ち、同じ末尾条件 `C` を持つ履歴を比較します。

\[
H_1=(A,B,C),\qquad H_2=(B,A,C)
\]

再帰更新を

\[
U_A(z)=az+u,\qquad
U_B(z)=az-u,\qquad
U_C(z)=cz
\]

とすると、

\[
z_{ABC}=c(a-1)u,\qquad
z_{BAC}=c(1-a)u
\]

であり、規則分離の閉形式は

\[
D_q=\lVert q_{ABC}-q_{BAC}\rVert
=2c(1-a)u\lVert d\rVert
\]

です。タスク数と直前条件が同じでも、更新写像の順序が異なるため規則目標が分離します。

#### 2. 状態一致介入と再構成

履歴依存の遅い規則目標を

\[
q_h=b+d z_h
\]

とし、共通介入で速い状態と作業規則を同時にリセットします。

\[
do(m_T=0,\theta_T=0)
\]

その後の規則再構成は

\[
\theta_{h,k+1}
=\theta_{h,k}+\gamma(q_h-\theta_{h,k})
\]

です。回復率は閉形式で

\[
R_q
=1-\frac{1}{K}\sum_{k=1}^{K}(1-\gamma)^k
\]

となります。

観測予測は

\[
p_{h,k}(y=1\mid x)=\sigma(\beta x^\top\theta_{h,k})
\]

で生成します。

### 三つの指標

#### SRRD（機構）

「高SRRD」は単一スカラーではなく、次のAND Gateです。

\[
D_q\ge1.00,
\qquad
R_q\ge0.70
\]

前者だけでは単なる履歴差、後者だけでは順序非依存の修復なので、両方を必須とします。

#### SRIV（力学的現象）

共通介入後、独立な等方診断プローブ上で Bernoulli 予測分布の Jensen–Shannon divergence を時間平均します。

\[
\mathrm{SRIV}
=\frac{1}{K}\sum_{k=1}^{K}
\mathbb E_{x\sim Q_D}
\left[
\operatorname{JSD}
\bigl(p_{ABC,k}(\cdot\mid x),p_{BAC,k}(\cdot\mid x)\bigr)
\right]
\]

#### CVP（観測量）

診断プローブとは別の held-out viability データで、再構成ありモデルが no-recovery ablation に対してどれだけ log loss を減らしたかを測ります。

\[
\mathrm{CVP}
=\frac{L_{\mathrm{ablated}}-L_{\mathrm{full}}}
{L_{\mathrm{ablated}}}
\]

CVP はラベルと予測値だけで算出され、潜在規則にはアクセスしません。

### 高SRRD・低CVPの構成的反例

直交反例では履歴依存規則を第2軸に限定し、CVPの評価入力を第1軸に限定します。

\[
q_h=(0,q_{h,2}),
\qquad
x_V=(x_1,0)
\]

したがって、規則がどれほど強く分離・再構成されても、

\[
x_V^\top\theta_{h,k}=0
\quad\Rightarrow\quad
p(y=1\mid x_V)=0.5
\]

です。これは検出力不足ではなく、履歴依存規則が CVP 観測写像の核に入ることによる厳密な反例です。一方、等方診断プローブは第2軸を観測するため SRIV は陽性になります。

### 確証実行結果

平均値。信頼区間は seed を単位とした percentile bootstrap 95% CI（4,000 resamples）です。

| Scenario | 規則分離 | 再構成率 | SRIV | CVP | 判定 |
|---|---:|---:|---:|---:|---|
| Aligned positive | 1.6185 | 0.8335 | 0.0415 | 0.5495 | 高SRRD・高CVP |
| Orthogonal counterexample | 2.6965 | 0.8339 | 0.2728 | 0.0000 | **高SRRD・低CVP** |
| Order-invariant viable | 0.0000 | 0.8334 | 0.0000 | 0.5949 | 低SRRD・高CVP |
| History shuffle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 陰性対照 |
| Frozen rule | 1.6251 | 0.0000 | 0.0000 | 0.0000 | 分離のみ・再構成なし |
| Flat state matched | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 陰性対照 |

#### Preregistered Gates

| Gate | 条件 | 結果 |
|---|---|---|
| G0 | 状態一致誤差 ≤ 1e-12 | PASS |
| G1 | aligned が高SRRD | PASS |
| G2 | orthogonal が高SRRD | PASS |
| G3 | orthogonal の SRIV ≥ 0.02 | PASS |
| G4 | orthogonal の CVP が ±0.02 内 | PASS |
| G5 | aligned の CVP ≥ 0.10 | PASS |
| G6 | 低SRRD・高CVPの逆向き分離 | PASS |
| G7 | frozen rule が再構成に失敗 | PASS |

閉形式の独立再計算では、規則分離誤差 `6.66e-16`、再構成率誤差 `2.22e-16`、直交反例の CVP 誤差 `0`、状態一致誤差 `0` でした。

### 科学的判定

この Phase 1 が示したのは SRRD の実在証明ではありません。示したのは次の三点です。

1. SRRD、SRIV、CVP は数学的に同じ量ではない。
2. 高SRRDでも、価値写像への結合がゼロならCVPはゼロになりうる。
3. 高CVPでも、履歴順序依存性がなければ高SRRDとはいえない。

よって CVP を SRRD の必要条件・十分条件・単独代理指標として使用してはいけません。実系で SRRD を支持するには、状態一致、履歴シャッフル、観測結合の同定、等価容量 RNN / Adaptive PSR との比較を別途通過する必要があります。

### 再現方法

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_phase1.py --output results --seeds 400 --bootstrap-replicates 4000
python scripts/verify_results.py --results results
```

主な出力:

- `results/phase1_seed_metrics.csv.gz`: 全2,400 seed の確証実行スナップショット（再実行時は非圧縮CSVを生成）
- `results/phase1_summary.csv`: 平均・95% CI
- `results/phase1_trajectories.csv`: 回復軌道
- `results/phase1_gates.json`: 事前登録 Gate 判定
- `results/phase1_analytic_checks.json`: 閉形式照合
- `preregistration/phase1.yaml`: 固定した仮説・閾値・反証条件

### 次段階

Phase 2 では、潜在規則を直接参照できないブラックボックス条件へ移し、以下を主反証にします。

- 等価容量 RNN / Adaptive PSR / History-MPC との held-out intervention 比較
- viability coupling を未知パラメータとして同定する回転プローブ
- state matching の等価性検定と residual imbalance 監査
- SRIV が CVP を予測しない領域の事前登録
- 合成系ではなく外部データでの再現
