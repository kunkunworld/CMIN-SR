# 论文结构建议

这份包的定位是：你上传到 Overleaf 后，可以直接从 `main.tex` 开始写。

## 推荐标题

Validity-Aware Spectral Diagnostics for Multifractal Random Walk Inference Under Finite Samples

中文理解：

有限样本下 MRW 推断的 validity-aware spectral diagnostic framework。

## 文章主线

不要把文章写成“我们训练了一个模型准确预测 lambda2”。

应该写成：

1. MRW-like spectral inference 在有限样本下很容易被误判；
2. 稳定 scaling 不等于 multifractal curvature；
3. 所以需要把问题拆成：
   - raw signal 到 empirical zeta；
   - empirical zeta 到 mono/MRW projection；
   - projection residual 和 calibrated diagnostics；
   - finite-sample identifiability limitation；
4. analytic zeta-space 的 spectral geometry 是可学习的；
5. raw time-series 接回来失败，说明瓶颈是 empirical zeta estimation；
6. deterministic estimator 也恢复不了 short-window lambda2，说明这是 finite-sample identifiability limitation；
7. 最终 CMIN-SR 是 diagnostic framework，不是 guaranteed lambda2 recovery engine。

## 主文建议结构

### 1. Introduction

讲问题，不讲太多模型细节。

重点句：

CMIN-SR should be interpreted as a validity-aware spectral diagnostic framework, not as a guaranteed short-window lambda2 recovery engine.

### 2. Framework

放 `fig1_cmin_sr_framework`。

讲：

- empirical zeta
- monofractal projection
- MRW projection
- residuals
- p_scaling / p_curved / p_mono / p_MRW
- tail instability

必须强调：

lambda2_proj is a projection coordinate, not proof of a true MRW mechanism.

### 3. Methods

讲 spectral geometry calibrator、projection diagnostics、finite-sample estimator study。

不要展开所有历史版本。

### 4. Experiments

主文放四组：

1. analytic spectral geometry calibration；
2. controlled process family diagnostics；
3. monofractal vs MRW projection / boundary；
4. finite-sample identifiability。

real-world sanity check 如果你想放，就放最后很短一段，或者 appendix。

### 5. Results

建议结果顺序：

1. analytic calibrator works；
2. process family diagnostics show caution is necessary；
3. ablation/history shows why each decomposition was needed；
4. deterministic estimators fail to recover lambda2 at short T。

### 6. Failure Analysis

这是文章的亮点，不要怕负结果。

讲：

- zeta noise bridge；
- analytic vs deterministic vs neural failure attribution；
- raw zeta quality is bottleneck。

### 7. Discussion

讲限制：

- q-grid sparse；
- scale range limited；
- high-q moment instability；
- short T under-identification；
- future: wavelet leaders / MFDFA / better zeta estimator。

## 主文图建议

1. `fig1_cmin_sr_framework`
2. `fig2_spectral_geometry_calibration`
3. `fig3_process_family_map`
4. `fig4_mrw_vs_mono_projection`
5. `fig5_finite_sample_identifiability`
6. `fig6_zeta_noise_bridge` 或 `fig6_failure_attribution`

## 主文表建议

1. `table1_process_family_diagnostics`
2. `table2_mrw_mono_projection`
3. `table4_identifiability`

`table3_ablation` 更适合 appendix 或 main text 简短引用。

## Appendix 建议

放：

- Tiny CMIN
- CMIN-Robust
- CMIN-SR v1/v2/v3
- Boundary calibration
- Raw zeta alignment
- Curvature-preserving zeta alignment
- failure attribution details

## 不要写的 claim

- 不要写 guaranteed lambda2 recovery。
- 不要写 lambda2_proj proves MRW。
- 不要把 real-world forecasting 写成核心成功标准。
- 不要说 negative result 是失败；它是 finite-sample identifiability insight。

