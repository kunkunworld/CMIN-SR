# CMIN-SR 项目实施与概念说明书

这份文档是给项目作者自己看的“总地图”。它不替代论文，也不替代各阶段实验日志，而是回答三个问题：

1. 这个项目到底在研究什么？
2. 代码和实验是如何一步步实现这个想法的？
3. 写论文和复现实验时，应该抓住哪条主线，哪些内容只放附录或失败分析？

本文档对应当前仓库状态。项目根目录为：

```text
D:/graduate_first/research/MS
```

论文工作区为：

```text
D:/graduate_first/research/MS/paper_writing_workspace
```

---

## 1. 一句话总结

CMIN-SR 是一个用于有限样本随机信号的 **validity-aware spectral diagnostic framework**。

它不是一个保证从短时间序列中精确恢复 MRW 参数 `lambda2` 的估计器，也不是一个证明真实数据服从 MRW 机制的模型。它更合理的定位是：

```text
在有限样本、多尺度统计噪声和高阶矩不稳定条件下，
组织 empirical spectrum、monofractal projection、MRW projection、
residual geometry、curvature、tail instability 等证据，
给出保守的谱几何诊断。
```

最终论文里最重要的一句话应该是：

```text
CMIN-SR should be interpreted as a validity-aware spectral diagnostic framework,
not as a guaranteed short-window lambda2 recovery engine.
```

---

## 2. 项目研究的问题

复杂系统、金融波动、湍流、生理信号等时间序列经常表现出多尺度波动和可能的多重分形结构。传统多重分形分析通常会从结构函数或其他多尺度统计量中估计：

- `zeta(q)`：scaling exponent function；
- `alpha(q)`：局部奇异性强度；
- `f(alpha)`：多重分形谱；
- `H`：Hurst / monofractal scaling 坐标；
- `lambda2`：MRW 模型中的曲率 / intermittency 坐标。

问题在于，真实研究中样本往往有限，尺度范围有限，高 q 阶矩容易不稳定。于是一个本来是线性的 monofractal 信号，比如 fGn 或 Gaussian，也可能因为 finite-sample noise 看起来有“假曲率”。反过来，一个真正由 MRW 生成的样本，在短窗口里也可能看不出稳定曲率。

所以本项目的核心问题不是简单地问：

```text
这个信号是不是 MRW？
```

而是问：

```text
在当前样本长度、q-grid、scale range 和 estimator 下，
经验谱提供了多少支持 monofractal / MRW / boundary / unstable 的证据？
这些证据是否足够稳定，还是只是有限样本伪影？
```

---

## 3. 核心概念

### 3.1 Empirical spectrum: `zeta_emp(q)`

`zeta_emp(q)` 是从原始时间序列估计出来的经验尺度谱。直观上，它描述不同阶数 `q` 的波动量如何随尺度变化。

它是整个框架中最脆弱的一环：

```text
raw signal x(t)
  -> structure functions S_q(a)
  -> log-log slope over scales
  -> zeta_emp(q)
```

有限样本、尺度选择、高 q 矩不稳定、尾部扰动、regime switching 都会污染 `zeta_emp(q)`。

相关代码：

- `src/mrw_inverse/models/empirical_spectrum.py`
- `src/mrw_inverse/models/robust_zeta_estimator.py`

---

### 3.2 Monofractal projection

Monofractal 线性谱写作：

```text
zeta_mono(q) = q H
```

如果一个经验谱基本是线性的，那么它更适合被 monofractal 解释。项目中会把 `zeta_emp(q)` 投影到这个线性族上，得到：

- `H_mono`
- `zeta_mono(q)`
- `mono_residual_norm`

直觉：

```text
mono_residual_norm 小
=> 线性 monofractal 解释已经足够好
=> 不应该轻易说它是 MRW curved spectrum
```

相关代码：

- `src/mrw_inverse/models/monofractal_projection.py`

---

### 3.3 MRW projection

项目采用的 MRW 解析谱族为：

```text
zeta_MRW(q) = qH - 0.5 * lambda2 * q * (q - 2)
```

其中：

- `H` 控制整体线性斜率；
- `lambda2` 控制谱的二次曲率；
- `lambda2 = 0` 时接近 monofractal 边界；
- `lambda2` 越大，理论曲率越明显。

项目会把 `zeta_emp(q)` 投影到 MRW 二参数族上，得到：

- `H_proj`
- `lambda2_proj`
- `zeta_mrw(q)`
- `residual_norm`

非常重要：

```text
lambda2_proj 是 projection coordinate，不是真实机制证明。
```

也就是说，`lambda2_proj > 0` 只说明“这个经验谱在 MRW 族上的投影坐标有曲率”，不等于真实数据生成机制就是 MRW。

相关代码：

- `src/mrw_inverse/models/mrw_projection.py`

---

### 3.4 Residual geometry

项目真正强调的不是单个分数，而是多组证据一起看：

```text
MRW residual         = zeta_emp 与 zeta_mrw 的距离
mono residual        = zeta_emp 与 zeta_mono 的距离
MRW-vs-mono gain     = monofractal residual 被 MRW residual 改善了多少
boundary score       = 是否处于低 lambda2 / mono-MRW 边界
tail instability     = 高 q / heavy-tail 是否不稳定
```

典型解释：

```text
如果 MRW residual < mono residual 且 p_curved 高，
才比较支持 MRW-like curved spectrum。

如果 mono residual <= MRW residual 且 p_mono 高，
则更像 stable monofractal。

如果 tail instability 高，
则即使有 apparent curvature，也不能轻易解释为 MRW。
```

相关代码：

- `src/mrw_inverse/models/curvature_diagnostics.py`

---

### 3.5 五个诊断分数

项目中常见的分数不是“真实概率”，更准确地说是 calibrated diagnostic scores。

#### `p_scaling`

问题：

```text
这个信号是否存在稳定尺度律？
```

高 `p_scaling` 只说明有稳定 scaling，不说明有 MRW 多重分形曲率。

#### `p_curved`

问题：

```text
经验谱是否显著偏离线性 monofractal 谱？
```

这是 v3 之后引入的重点。它试图把“有没有曲率”从 “是不是 MRW” 中分离出来。

#### `p_mono`

问题：

```text
经验谱是否更适合 monofractal linear spectrum？
```

fGn / Gaussian 的理想表现是：

```text
p_scaling 高
p_curved 低
p_mono 高
p_MRW 低或中低
```

#### `p_MRW`

问题：

```text
经验谱是否稳定、弯曲，并且能被 MRW 二参数族合理解释？
```

它不能单独解释。必须和 `p_curved`, `p_mono`, residuals, tail instability 一起看。

#### `p_boundary`

问题：

```text
是否处于 low-lambda2 MRW / monofractal 的边界区域？
```

low-lambda2 MRW 不是失败，而是边界案例。它理论上就不应该被强行判成 strong MRW。

---

## 4. 最终框架流程

最终主线可以画成：

```text
raw stochastic signal
  -> empirical zeta(q)
  -> monofractal projection
  -> MRW projection
  -> residual / curvature / boundary / instability diagnostics
  -> calibrated diagnostic scores
  -> finite-sample identifiability interpretation
```

更细一点：

```text
Stage 1: Raw signal to empirical spectrum
  x(t)
  -> structure functions S_q(a)
  -> zeta_emp(q)

Stage 2: Projection geometry
  zeta_emp(q)
  -> zeta_mono(q), H_mono, mono_residual
  -> zeta_MRW(q), H_proj, lambda2_proj, MRW residual

Stage 3: Diagnostics
  residuals + curvature + boundary + instability
  -> p_scaling, p_curved, p_mono, p_MRW, p_boundary

Stage 4: Interpretation
  conservative evidence organization
  not mechanism proof
```

---

## 5. 代码结构总览

### 5.1 核心模型目录

```text
src/mrw_inverse/models/
```

主要文件：

- `empirical_spectrum.py`
  - 基础经验谱估计工具。
- `robust_zeta_estimator.py`
  - deterministic structure-function estimator；
  - 包括 OLS、trimmed、bootstrap、smoothed 等有限样本估计。
- `mrw_projection.py`
  - MRW 二参数投影。
- `monofractal_projection.py`
  - 线性 monofractal 投影。
- `curvature_diagnostics.py`
  - 曲率、线性度、MRW-vs-mono gain、boundary score 等诊断。
- `spectral_geometry_calibrator.py`
  - 小型 spectrum-space calibrator；
  - 输入 zeta / residual / diagnostics；
  - 输出 `p_scaling`, `p_curved`, `p_mono`, `p_MRW`, `p_boundary`。
- `spectral_representation_model.py`
  - CMIN-SR 历史模型主体。
- `zeta_aligned_encoder.py`
  - raw signal 到 `zeta_emp(q)` 的 zeta alignment 模型。

---

### 5.2 数据集目录

```text
src/mrw_inverse/data/
```

主要文件：

- `analytic_spectrum_dataset.py`
  - 直接生成解析 `zeta(q)` 曲线；
  - 用于训练 spectral geometry calibrator；
  - 不依赖 raw time series。
- `raw_zeta_alignment_dataset.py`
  - 生成 raw time series，同时提供理论 zeta target；
  - 用于训练 raw zeta alignment。
- `boundary_calibration_dataset.py`
  - same-H fGn/MRW lambda2 sweep；
  - 用于 boundary calibration ablation。

---

### 5.3 损失函数目录

```text
src/mrw_inverse/losses/
```

主要文件：

- `spectral_representation_losses.py`
  - 原始 CMIN-SR / v1-v3 相关 loss。
- `spectrum_space_calibration_losses.py`
  - analytic zeta 空间的分类、排序、边界、caution loss。
- `zeta_alignment_losses.py`
  - raw zeta alignment loss。
- `curvature_preserving_zeta_losses.py`
  - band-specific MRW curvature preservation；
  - third-difference smoothness；
  - lambda2 projection consistency；
  - MRW-vs-mono residual margin。

---

### 5.4 分析模块

```text
src/mrw_inverse/analysis/
```

主要文件：

- `curvature_identifiability.py`
  - deterministic estimator-level lambda2 recovery study；
  - 比较 OLS / trimmed / bootstrap / smoothed structure-function estimators；
  - 输出 `lambda2_proj`, residuals, curvature scores, instability warnings。

---

## 6. 实验阶段如何演化

这个项目不是一步到位，而是通过多轮失败和校正逐渐清楚主线的。下面按阶段解释。

### 6.1 Proxy / Tiny CMIN / CMIN-Robust

早期阶段主要是在确认：

- 原始网络是否容易把 stable scaling 都误判成 MRW；
- negative controls 是否有用；
- robust / anti-confounding loss 是否能压低明显非 MRW 的假阳性。

这些阶段现在主要作为 legacy / appendix，不建议放主文展开。

相关文档：

- `docs/legacy_experiments_map.md`
- `docs/cmin_robust_training.md`
- `docs/cmin_robust_multilength.md`

---

### 6.2 CMIN-SR v1

问题：

```text
模型容易把 fGn 的 stable scaling 也解释成 MRW-compatible。
```

原因：

```text
p_MRW 同时承担了 scaling stability 和 MRW compatibility 两个语义。
```

结果：

- fGn 的 `p_MRW` 过高；
- 模型会把“稳定尺度律”误读成“MRW-like”。

---

### 6.3 CMIN-SR v2: monofractal competition

v2 加入 monofractal projection，让 MRW 和 linear monofractal 竞争。

目标：

```text
如果 linear monofractal 已经解释得很好，
就不应该给很高 p_MRW。
```

效果：

- fGn / Gaussian 的 `p_MRW` 降低；
- 但 MRW 自身的 `p_MRW` 也被压低。

结论：

```text
v2 不是失败，但它像是整体压低水位，
还没有真正区分 linear stable scaling 和 curved MRW scaling。
```

相关文档：

- `docs/cmin_sr_v2_monofractal_calibration.md`

---

### 6.4 CMIN-SR v3: explicit p_curved

v3 引入 `p_curved`，把问题拆成：

```text
p_scaling: 是否有稳定尺度律？
p_curved: 是否有显著曲率？
p_mono: 是否更适合线性 monofractal？
p_MRW: 是否是稳定、显著弯曲、MRW-compatible？
```

但结果仍是 partial success：

- 工程链路跑通；
- `p_curved` head 存在；
- 但 fGn 和 MRW 的 `p_curved / p_MRW` 仍太接近。

结论：

```text
仅仅加一个 head 不够。
问题不只是输出层，而是 raw zeta_emp 本身带有有限样本伪曲率。
```

相关文档：

- `docs/cmin_sr_v3_curved_linear_calibration.md`

---

### 6.5 Boundary Calibration: same-H contrast

这一阶段尝试 same-H fGn/MRW(lambda2 sweep) 对比训练：

```text
同一个 H 下：
fGn(H)
MRW(H, lambda2=0.01)
MRW(H, lambda2=0.03)
MRW(H, lambda2=0.06)
MRW(H, lambda2=0.10)
```

目标：

```text
让模型学会 stable scaling 不等于 curved MRW。
```

结果：

- raw time-series contrastive fine-tuning 仍不足；
- fGn 和 MRW 仍然太像。

结论：

```text
raw finite-sample spectrum noise 可能淹没了弱曲率信号。
```

相关文档：

- `docs/cmin_sr_boundary_calibration.md`

---

### 6.6 Spectrum-Space Calibration

这是关键转折。

做法：

```text
不从 raw time series 学，
而是在干净或可控扰动的 analytic zeta(q) 空间里训练 calibrator。
```

数据类型包括：

- `linear_mono`
- `boundary_mrw`
- `curved_mrw`
- `noisy_mono`
- `noisy_mrw`
- `heavy_tail_distorted`
- `regime_apparent`
- `ambiguous_mild_curvature`

结果很清楚：

- analytic spectrum-space calibrator 能区分 linear / boundary / curved / unstable；
- lambda2 sweep 的 monotonicity 可以达到 1.0；
- heavy-tail / regime 可以被压低 MRW compatibility。

结论：

```text
谱几何解释层是可学的。
瓶颈不是 p_MRW 定义，也不是缺 head。
瓶颈是 raw CMIN-SR 输出的 zeta_emp 太脏。
```

相关代码：

- `src/mrw_inverse/data/analytic_spectrum_dataset.py`
- `src/mrw_inverse/models/spectral_geometry_calibrator.py`
- `src/mrw_inverse/losses/spectrum_space_calibration_losses.py`

相关文档：

- `docs/spectrum_space_calibration.md`
- `docs/spectral_geometry_calibrator.md`

---

### 6.7 Raw Zeta Alignment

下一步修 raw signal -> zeta_emp。

目标：

```text
先让 raw encoder 学会输出更干净的 zeta_emp(q)，
再交给 spectral geometry calibrator 解释。
```

结果：

- fGn / Gaussian 的 fake curvature 被明显压低；
- 但是 MRW 的真实曲率也被压掉。

典型结论：

```text
raw zeta alignment 成功把谱变线性，
但太保守，把 MRW 曲率也 over-linearized。
```

相关代码：

- `src/mrw_inverse/data/raw_zeta_alignment_dataset.py`
- `src/mrw_inverse/models/zeta_aligned_encoder.py`
- `src/mrw_inverse/losses/zeta_alignment_losses.py`

相关文档：

- `docs/raw_zeta_alignment.md`
- `docs/zeta_alignment_comparison.md`

---

### 6.8 Curvature-Preserving Zeta Alignment

为修复 over-linearization，引入：

- MRW lambda2 band-specific curvature loss；
- high-lambda MRW 更强二阶曲率匹配；
- third-difference smoothness，而不是二阶差分 flattening；
- lambda2 projection consistency；
- MRW-vs-mono residual margin。

结果：

- fGn fake curvature 能保持被压低；
- 但 MRW medium/high curvature 仍不稳定；
- `lambda2_true` vs `lambda2_proj` correlation 仍接近 0。

结论：

```text
这不是继续加 head 能解决的问题。
可能是当前 q-grid / scale range / T 下 empirical curvature 本身不可可靠识别。
```

相关代码：

- `src/mrw_inverse/losses/curvature_preserving_zeta_losses.py`

相关文档：

- `docs/curvature_preserving_zeta_alignment.md`
- `docs/curvature_preserving_zeta_comparison.md`

---

### 6.9 Finite-Sample Curvature Identifiability Study

最后一个关键阶段：不再训练新模型，而是问：

```text
如果用 deterministic / oracle-like structure-function estimator，
能不能从 raw MRW samples 恢复 lambda2？
```

比较 estimators：

- `structure_ols`
- `structure_trimmed`
- `structure_bootstrap`
- `structure_smoothed`

测试：

- 不同 T；
- 不同 q-grid；
- 不同 scale range；
- analytic zeta -> noisy zeta -> raw zeta bridge；
- analytic / deterministic / neural failure attribution。

核心结果：

```text
在当前短窗口设置下，deterministic estimators 也不能稳定恢复 lambda2。
```

这说明 CMIN-SR raw 端失败主要是：

```text
finite-sample empirical spectrum estimation bottleneck
```

而不是：

```text
网络太小 / head 不够 / loss 不够复杂
```

相关代码：

- `src/mrw_inverse/analysis/curvature_identifiability.py`
- `experiments/run_finite_sample_curvature_identifiability.py`
- `experiments/run_scale_length_sensitivity.py`
- `experiments/run_qgrid_sensitivity.py`
- `experiments/run_zeta_noise_bridge.py`
- `experiments/run_cmin_sr_failure_attribution.py`

相关文档：

- `docs/finite_sample_curvature_identifiability.md`
- `docs/estimator_curvature_recovery.md`
- `docs/zeta_noise_bridge.md`
- `docs/cmin_sr_failure_attribution.md`

---

## 7. 最终论文主线怎么讲

推荐主线：

```text
1. 多重分形谱估计在有限样本下很难。
2. 单纯从 empirical zeta 曲率判断 MRW 很危险。
3. CMIN-SR 将问题拆成：
   empirical spectrum estimation
   + monofractal/MRW projection
   + spectral geometry calibration
   + instability warning。
4. 在 clean analytic zeta 空间中，linear/boundary/curved/unstable 谱几何是可区分的。
5. 但在 raw finite samples 中，zeta_emp 质量成为瓶颈。
6. deterministic estimator 也无法稳定恢复 lambda2，说明这是有限样本 identifiability limitation。
7. 因此 CMIN-SR 的合理价值是诊断框架，而不是保证恢复 MRW 参数。
```

---

## 8. 哪些结果放主文，哪些放附录

### 8.1 建议主文

1. Spectral geometry calibration
   - 证明解释层在 clean zeta 空间是有效的。
2. Process-family diagnostics
   - 展示 fGn / Gaussian / Student-t / GARCH / RegimeSwitch 不应被简单过度解释成 MRW。
3. Monofractal vs MRW projection
   - 展示 residual geometry 的必要性。
4. Finite-sample curvature identifiability
   - 论文最有价值的负结果 / limitation。
5. Zeta noise bridge / failure attribution
   - 解释为什么 analytic calibrator 接回 raw zeta 会失败。

### 8.2 建议附录

1. Tiny CMIN / CMIN-Robust 早期版本。
2. CMIN-SR v1/v2/v3 发展过程。
3. Boundary calibration negative result。
4. Raw zeta alignment trade-off。
5. Curvature-preserving zeta alignment trade-off。
6. 完整 estimator-level tables。
7. real-data sanity check。

### 8.3 不建议主文强讲

- “模型证明某真实数据服从 MRW”。
- “短窗口下能保证恢复 lambda2”。
- “CMIN-SR outperform 所有经典多重分形估计器”。
- “p_MRW 是真实机制概率”。

---

## 9. 如何复现实验

开发环境中通常使用：

```bash
conda run -n for_codex python <script.py>
```

或者进入环境：

```bash
conda activate for_codex
python <script.py>
```

### 9.1 快速检查

```bash
python scripts/smoke_test_paper_pipeline.py
```

### 9.2 生成论文资产

```bash
python experiments/paper/generate_all_paper_assets.py --quick
```

### 9.3 运行主实验 wrapper

```bash
python experiments/paper/run_exp1_spectral_geometry_calibration.py --quick
python experiments/paper/run_exp2_process_family_diagnostics.py --quick
python experiments/paper/run_exp3_boundary_projection.py --quick
python experiments/paper/run_exp4_finite_sample_identifiability.py --quick
python experiments/paper/run_exp5_real_world_sanity_check.py --quick
```

### 9.4 有限样本 identifiability 实验

```bash
python experiments/run_finite_sample_curvature_identifiability.py --quick
python experiments/run_scale_length_sensitivity.py --quick
python experiments/run_qgrid_sensitivity.py --quick
python experiments/run_zeta_noise_bridge.py --quick
python experiments/run_cmin_sr_failure_attribution.py --quick
```

### 9.5 论文 LaTeX 编译

进入：

```text
paper_writing_workspace/
```

使用 VS Code LaTeX Workshop 或 MiKTeX 编译：

```text
pdflatex -> bibtex -> pdflatex -> pdflatex
```

主文件：

```text
paper_writing_workspace/main.tex
```

---

## 10. 重要输出目录

### 10.1 模型 checkpoint

```text
checkpoints/cmin/
```

重要 checkpoint：

- `cmin_sr_synthetic.pt`
- `cmin_sr_v2_synthetic.pt`
- `cmin_sr_v3_synthetic.pt`
- `cmin_sr_calibrated_synthetic.pt`
- `spectral_geometry_calibrator.pt`
- `cmin_sr_zeta_aligned.pt`
- `cmin_sr_zeta_curvature_preserved.pt`

### 10.2 实验输出

```text
outputs/reports/
outputs/tables/
outputs/figures/
```

### 10.3 论文资产

```text
paper_assets/
paper_writing_workspace/paper_assets/
```

最近整理过的绘图数据在：

```text
paper_writing_workspace/paper_assets/figure_data/
```

图表审稿报告在：

```text
paper_writing_workspace/paper_assets/summaries/
```

---

## 11. 当前论文写作状态

LaTeX 工作区：

```text
paper_writing_workspace/
```

主文件：

```text
main.tex
```

章节：

- `sections/01_introduction.tex`
- `sections/02_framework.tex`
- `sections/03_methods.tex`
- `sections/04_experiments.tex`
- `sections/05_results.tex`
- `sections/06_failure_analysis.tex`
- `sections/07_discussion.tex`
- `sections/08_conclusion.tex`

辅助文件：

- `references.bib`
- `latex_tables/`
- `figures/`
- `paper_assets/`
- `ARTICLE_STRUCTURE_CN.md`
- `paper_draft.md`
- `claim_evidence_table.md`
- `TODO_missing_evidence.md`
- `reviewer2_comments.md`

---

## 12. 图表整理建议

最近审稿整理的结论：

### 必须重画

- Fig.1 pipeline
  - 当前太像临时流程图；
  - 应改成三阶段 pipeline：
    1. finite increments -> empirical zeta estimation；
    2. mono/MRW projection and residual geometry；
    3. calibrated diagnostics with warning。

### 建议优先重画

- Fig.5 finite-sample lambda2 recovery
  - 应成为正文核心主图；
  - 比大表更能表达 finite-sample limitation。
- Fig.4 spectral geometry map
- Fig.7 failure attribution
- Fig.8 projection residual geometry

### 可以放附录或弱化

- Fig.9 real-data sanity check
  - 适合作为 sanity check；
  - 不适合作为强 real-world validation。

相关报告：

- `paper_writing_workspace/paper_assets/summaries/top_journal_figure_table_audit.md`
- `paper_writing_workspace/paper_assets/summaries/top_journal_writing_audit.md`
- `paper_writing_workspace/paper_assets/summaries/latex_float_and_layout_fix_report.md`

---

## 13. 常见误解和正确说法

### 误解 1：`p_MRW` 高就证明数据是 MRW

错误。

正确说法：

```text
p_MRW is a calibrated diagnostic compatibility score.
It should be interpreted together with p_curved, p_mono, residuals,
tail instability, and finite-sample uncertainty.
```

### 误解 2：`lambda2_proj` 就是真实 `lambda2`

错误。

正确说法：

```text
lambda2_proj is a projection coordinate.
It does not prove the data-generating mechanism.
```

### 误解 3：模型失败了，所以论文价值不大

不准确。

项目最有价值的结论恰恰是：

```text
clean spectral geometry is identifiable,
but raw finite-sample curvature recovery is bottlenecked by empirical zeta estimation.
```

这使论文从“刷指标模型”变成“有限样本多重分形诊断与可识别性分析”。

### 误解 4：继续加 head / loss / backbone 就能解决

目前证据不支持。

deterministic estimators 也恢复不了 `lambda2`，说明瓶颈更可能在：

- q-grid 太稀；
- scale range 太窄；
- T 太短；
- high-q moments 不稳定；
- structure-function estimator 不够强；
- MRW 曲率在有限样本下本身不可识别。

---

## 14. 下一步最合理的工作

如果目标是写论文，而不是继续研发模型，建议顺序：

1. 重画 Fig.1。
2. 重画 Fig.5，把 finite-sample identifiability 讲清楚。
3. 精修 Introduction 和 Conclusion，强调 diagnostic framework。
4. 把 legacy 阶段移到 appendix 或 supplement。
5. 检查所有 claim-evidence 对应关系。
6. 准备投稿版本的表格和 caption。
7. 经典 estimator baseline 已补充到 quick-grid supplement：包括 aggregate
   structure function、MFDFA、一阶/二阶 MFDFA、Haar wavelet leader 和
   Haar WTMM-style control。注意：Haar wavelet rows 是轻量控制，不是完整
   production WTMM / wavelet-leader 软件包。

---

## 15. 最短读项目路线

如果你只想快速重新理解项目，建议按这个顺序读：

1. `README.md`
2. `docs/project_implementation_guide_cn.md`
3. `docs/final_project_manifest.md`
4. `docs/spectral_representation_framework.md`
5. `docs/finite_sample_curvature_identifiability.md`
6. `paper_writing_workspace/ARTICLE_STRUCTURE_CN.md`
7. `paper_writing_workspace/paper_assets/summaries/top_journal_figure_table_audit.md`

如果你要改代码，再读：

1. `docs/core_code_index.md`
2. `src/mrw_inverse/models/mrw_projection.py`
3. `src/mrw_inverse/models/monofractal_projection.py`
4. `src/mrw_inverse/models/curvature_diagnostics.py`
5. `src/mrw_inverse/models/spectral_geometry_calibrator.py`
6. `src/mrw_inverse/analysis/curvature_identifiability.py`
7. `src/mrw_inverse/analysis/classical_multifractal_estimators.py`

---

## 16. 最终定位

这个项目最稳的论文定位是：

```text
一个物理/谱几何约束的有限样本多重分形诊断框架。
它显示：
1. analytic spectral geometry 可以被稳定学习；
2. raw empirical spectrum estimation 是主要瓶颈；
3. short-window lambda2 recovery 在当前 estimator/q-grid/scale range 下不可可靠保证；
4. 因此多重分形诊断应同时报告 scaling、curvature、mono/MRW residuals、
   instability warnings 和 finite-sample identifiability limits。
```

这条主线保守、清楚，也最不容易被审稿人攻击。
