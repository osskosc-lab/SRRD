# SRRD Phase 1 Falsification

SRRD → SRIV → CVP を「三つの競合理論」ではなく、**原理 → 力学的現象 → 観測量**として分離し、相互の同一視を反証するための最小合成実験です。

## 結論

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

## 三層の独立定義

この Phase 1 では、略称の意味よりも測定対象を固定します。SRIV は *State-matched Rule-induced Intervention Variance*、CVP は *Counterfactual Viability Preservation* として操作化しています。

| 層 | 対象 | 入力として許されるもの | 禁止するもの |
|---|---|---|---|
| SRRD | 履歴依存の規則分離と、介入後の規則再構成 | 潜在規則目標と規則回復軌道 | 予測値、ラベル、効用、CVP |
| SRIV | 状態一致後に生じる介入応答分布の差 | 診断プローブ上の対予測分布 | ラベル、効用、潜在規則 |
| CVP | 再構成あり／なしの反実仮想的な生存機能差 | 独立した評価データのラベルと予測値 | 履歴符号、潜在規則、SRRD、SRIV |

実装上も各量は別関数で計算し、引数を共有しません。

## 最小数理モデル

### 1. 履歴順序の符号化

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

### 2. 状態一致介入と再構成

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

## 三つの指標

### SRRD（機構）

「高SRRD」は単一スカラーではなく、次のAND Gateです。

\[
D_q\ge1.00,
\qquad
R_q\ge0.70
\]

前者だけでは単なる履歴差、後者だけでは順序非依存の修復なので、両方を必須とします。

### SRIV（力学的現象）

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

### CVP（観測量）

診断プローブとは別の held-out viability データで、再構成ありモデルが no-recovery ablation に対してどれだけ log loss を減らしたかを測ります。

\[
\mathrm{CVP}
=\frac{L_{\mathrm{ablated}}-L_{\mathrm{full}}}
{L_{\mathrm{ablated}}}
\]

CVP はラベルと予測値だけで算出され、潜在規則にはアクセスしません。

## 高SRRD・低CVPの構成的反例

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

## 確証実行結果

平均値。信頼区間は seed を単位とした percentile bootstrap 95% CI（4,000 resamples）です。

| Scenario | 規則分離 | 再構成率 | SRIV | CVP | 判定 |
|---|---:|---:|---:|---:|---|
| Aligned positive | 1.6185 | 0.8335 | 0.0415 | 0.5495 | 高SRRD・高CVP |
| Orthogonal counterexample | 2.6965 | 0.8339 | 0.2728 | 0.0000 | **高SRRD・低CVP** |
| Order-invariant viable | 0.0000 | 0.8334 | 0.0000 | 0.5949 | 低SRRD・高CVP |
| History shuffle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 陰性対照 |
| Frozen rule | 1.6251 | 0.0000 | 0.0000 | 0.0000 | 分離のみ・再構成なし |
| Flat state matched | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 陰性対照 |

### Preregistered Gates

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

## 科学的判定

この Phase 1 が示したのは SRRD の実在証明ではありません。示したのは次の三点です。

1. SRRD、SRIV、CVP は数学的に同じ量ではない。
2. 高SRRDでも、価値写像への結合がゼロならCVPはゼロになりうる。
3. 高CVPでも、履歴順序依存性がなければ高SRRDとはいえない。

よって CVP を SRRD の必要条件・十分条件・単独代理指標として使用してはいけません。実系で SRRD を支持するには、状態一致、履歴シャッフル、観測結合の同定、等価容量 RNN / Adaptive PSR との比較を別途通過する必要があります。

## 再現方法

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_phase1.py --output results --seeds 400 --bootstrap-replicates 4000
python scripts/verify_results.py --results results
```

主な出力:

- `results/phase1_seed_metrics.csv`: 全2,400 seed
- `results/phase1_summary.csv`: 平均・95% CI
- `results/phase1_trajectories.csv`: 回復軌道
- `results/phase1_gates.json`: 事前登録 Gate 判定
- `results/phase1_analytic_checks.json`: 閉形式照合
- `preregistration/phase1.yaml`: 固定した仮説・閾値・反証条件

## 次段階

Phase 2 では、潜在規則を直接参照できないブラックボックス条件へ移し、以下を主反証にします。

- 等価容量 RNN / Adaptive PSR / History-MPC との held-out intervention 比較
- viability coupling を未知パラメータとして同定する回転プローブ
- state matching の等価性検定と residual imbalance 監査
- SRIV が CVP を予測しない領域の事前登録
- 合成系ではなく外部データでの再現

