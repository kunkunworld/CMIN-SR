# CMIN-SR 理论多智能体审查报告

本报告由三个审查角色共同完成：

1. **数学一致性审查员**：重点检查公式、定义、投影、结构函数、`zeta(q)`、`alpha(q)`、`f(alpha)`、`lambda2_proj` 是否与代码一致。
2. **多重分形 / MRW 建模审稿人**：重点检查理论主线是否符合复杂系统、多重分形和 CSF 风格论文要求。
3. **Claim-Evidence / 幻觉审查员**：重点检查是否存在过强 claim、证据不足、引用不支撑或容易被审稿人攻击的表述。

结论先说：

```text
项目理论主线总体站得住，但必须保持保守定位。

CMIN-SR 最稳的定义是：
spectral-geometry-constrained finite-sample diagnostic framework。

它不应被写成：
1. 保证短窗口恢复 lambda2 的估计器；
2. 证明真实数据服从 MRW 机制的方法；
3. 已严格统计校准的概率模型。
```

---

## 1. 总体判断

三位审查智能体的共识是：

- 当前项目的核心主线合理；
- 最有价值的贡献不是“神经网络成功恢复 MRW 参数”，而是：
  - 将多重分形诊断拆解为 empirical spectrum estimation 和 spectral geometry interpretation；
  - 证明 clean analytic spectrum space 中 linear / boundary / curved / unstable geometry 是可区分的；
  - 进一步指出 raw finite-sample `zeta_emp(q)` 是主要瓶颈；
  - 给出有限样本曲率可识别性的经验性限制。

论文最稳的说法是：

```text
CMIN-SR organizes finite-sample multifractal evidence through empirical spectra,
monofractal/MRW projections, residual geometry, curvature diagnostics,
boundary scores, and instability warnings.
```

不稳的说法是：

```text
CMIN-SR recovers the true MRW mechanism from short noisy samples.
```

---

## 2. 数学审查：最重要问题

### 2.1 Critical: 结构函数对象在文稿和代码之间不完全一致

这是三份审查中最重要的数学问题。

论文中当前理论设定大致是：

```text
输入 x_t 是 return / increment；
路径 X_t = sum x_t；
尺度 a 上的聚合增量为：

Delta_a X_t = sum_{i=0}^{a-1} x_{t+i}
```

这对应传统结构函数：

```text
S_q(a) = average |Delta_a X_t|^q
S_q(a) ~ a^{zeta(q)}
```

但是代码中有不同实现口径：

1. `empirical_spectrum.py`
   - 先 `path = cumsum(x)`；
   - 再取 path increment；
   - 与论文中的 aggregated increment 口径基本一致。

2. `robust_zeta_estimator.py`
   - 直接用 `x[:, scale:] - x[:, :-scale]`；
   - 如果 `x` 是 return / increment，这就变成了 return 的差分，而不是聚合 return。

3. 另一个前端模型中存在非重叠卷积求和；
   - 这又是第三种近似口径。

为什么这重要？

```text
如果不同 estimator 对 x_t 的解释不同，
那么它们估计的 zeta(q) 并不是完全同一个数学对象。

这会影响：
- zeta_emp(q)
- lambda2_proj
- finite-sample curvature recovery
- deterministic estimator comparison 的可比性
```

建议：

1. 在论文中明确写成：

```text
The intended structure-function object is based on aggregated increments
of the associated path.
```

2. 同时承认实现中有 overlapping / non-overlapping / deterministic baseline variants：

```text
Implementation variants differ in overlapping versus non-overlapping aggregation,
and deterministic baselines are interpreted as estimator-level diagnostics rather
than exact replicas of a single ideal estimator.
```

3. 如果后续要做代码级修复，优先统一 `robust_zeta_estimator.py` 的输入语义：

```text
如果输入是 returns，则先 cumsum 成 path，再取 path increments；
如果输入已经是 path，则直接取 path increments。
```

这不是必须马上改论文初稿的问题，但如果投稿前不解释，审稿人可能会抓住。

---

### 2.2 Major: least-squares projection 的表述比代码更理想化

论文中容易写成：

```text
(H_proj, lambda2_proj) = argmin over MRW family
```

但代码实际做法更接近：

```text
1. 在选定 q-grid 上做 least-squares coordinate fit；
2. 对 H 和 lambda2 做 clipping / admissibility projection；
3. 得到 zeta_mrw(q) 和 residual。
```

这不是严格意义上的 constrained least squares solver。

建议论文写法：

```text
We fit the monofractal and MRW families by least-squares coordinates on
the selected q-grid, followed by admissibility clipping of H and lambda2.
The resulting lambda2_proj is a projection coordinate, not a generative
parameter estimate.
```

这样既准确，也保守。

---

### 2.3 Major: `alpha(q)` / `f(alpha)` 是有限 q-grid 近似，不是完整谱

项目中有两种口径：

1. 对 MRW analytic decoder，可以解析求导；
2. 对 empirical spectrum / baseline，常用 `np.gradient` 做有限差分。

默认 q-grid 是：

```text
q = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

这意味着：

- 没有负 q；
- 没有 q=0；
- 只能看到正 q 区间的一段谱；
- `f(alpha)` 不是完整 singularity spectrum；
- 端点的 finite-difference derivative 会有偏差。

建议论文中明确：

```text
For empirical spectra, alpha(q) and f(alpha(q)) are finite-grid Legendre
approximations over the tested positive q-grid, not a full singularity-spectrum
reconstruction.
```

---

### 2.4 Major: MRW simulation 不应写成每个有限样本精确满足解析谱

MRW synthetic generator 的设计目标是生成与 MRW parabolic spectrum 一致的参考族，但有限样本中会受到：

- finite length；
- integral scale truncation；
- normalization；
- estimator noise；
- sample-path fluctuation；
- q-grid limitation；
- scale range limitation。

所以不能写成：

```text
synthetic MRW samples match the analytic projection family
```

更稳的写法：

```text
The synthetic MRW generator targets the parabolic MRW spectrum as a reference
family in the asymptotic/model-design sense, but finite length, integral-scale
truncation, normalization, and estimator noise mean that realized samples need
not exhibit the analytic curvature reliably.
```

这句话非常重要，因为它直接解释为什么 raw MRW curvature recovery 会失败。

---

### 2.5 Major: `p_curved` 检测的是非线性幅度，不单独保证凹性方向

代码中的 curvature diagnostics 使用二阶差分幅度：

```text
curvature_score ~ |D2 zeta|
```

这能检测“非线性”，但并不单独区分：

- concave curvature；
- convex curvature；
- jagged high-q distortion。

而标准 MRW parabolic spectrum 的二阶导数是：

```text
zeta''(q) = -lambda2 <= 0
```

所以论文中必须避免把 `p_curved` 直接等同于 “MRW curvature”。

正确说法：

```text
p_curved measures nonlinear departure from a linear monofractal spectrum.
MRW compatibility additionally requires MRW residual improvement,
appropriate projection geometry, boundary behavior, and instability checks.
```

---

## 3. 建模审查：理论主线是否合理

### 3.1 主线总体合理

建模审查认为，这个项目已经比较适合 CSF 风格：

- 不是普通 deep learning benchmark；
- 重点是 finite-size effects；
- 讨论了 spurious multiscaling；
- 引入了 heavy-tail / GARCH / regime-switching controls；
- 有 negative results 和 identifiability limitation。

这比“我们训练了一个网络恢复 lambda2”更有科学价值。

---

### 3.2 “physics-constrained” 要降温

项目确实有约束：

- MRW analytic spectrum；
- monofractal projection；
- residual geometry；
- boundary behavior；
- instability warnings。

但这些更准确地说是：

```text
spectral geometry constraints
```

而不是强物理定律，比如守恒律、偏微分方程、哈密顿结构等。

建议主文优先使用：

```text
spectral-geometry-constrained diagnostic inverse framework
```

少用：

```text
physics-constrained neural inverse problem
```

如果使用 physics-constrained，必须解释：

```text
这里的 physical constraint 指 MRW / multifractal cascade reference family
和可解释谱几何约束，而不是一般物理守恒律。
```

---

### 3.3 各 process family 的角色是清楚的

建议在 Methods 或 Experiments 中明确一张小表：

| Process | Role |
|---|---|
| MRW | reference positive / projection family |
| low-lambda2 MRW | boundary between monofractal and curved MRW |
| fGn | stable monofractal long-memory control |
| Gaussian | iid linear negative control |
| Student-t | heavy-tail / high-q instability control |
| GARCH | volatility clustering confounder |
| RegimeSwitch | nonstationary variance / apparent curvature confounder |

这样审稿人会更容易理解为什么不是只比较 MRW vs Gaussian。

---

### 3.4 negative result 可以写成贡献

当前结果链条是有逻辑的：

```text
analytic zeta space: geometry separable
raw zeta space: fGn/MRW separation fails
zeta alignment: fake curvature can be suppressed
curvature preservation: true generated-MRW curvature still hard
deterministic estimators: lambda2 recovery also weak
```

这支持结论：

```text
问题不只是神经网络，而是当前有限样本 empirical spectrum estimation bottleneck。
```

这可以作为 CSF 风格论文中的重要 scientific insight。

---

## 4. Claim-Evidence 审查

### 4.1 “finite-sample identifiability” 不能写得像数学定理

当前证据支持：

```text
under tested T, q-grid, scale range, and estimator family,
lambda2 recovery is weak.
```

但不支持：

```text
MRW lambda2 is theoretically unidentifiable in finite samples.
```

建议统一写成：

```text
empirical finite-sample identifiability limitation under the tested estimators
```

或者：

```text
finite-sample recovery limitation under the tested q-grid and scale range
```

---

### 4.2 “calibrated” 不能让人误解成统计概率校准

项目中 `p_scaling`, `p_curved`, `p_mono`, `p_MRW`, `p_boundary` 经常写成概率式分数。

但目前没有：

- reliability diagram；
- ECE；
- Brier score；
- posterior calibration；
- uncertainty quantification。

所以第一次出现 `calibrated score` 时应加说明：

```text
Here, calibrated denotes model-trained diagnostic normalization, not verified
statistical probability calibration.
```

建议少写：

```text
probability that the signal is MRW
```

多写：

```text
MRW-compatibility diagnostic score
```

---

### 4.3 analytic success 和 raw success 必须分开

项目确实证明：

```text
在 clean analytic zeta space 中，linear / boundary / curved / unstable spectra 可区分。
```

但 raw process-family table 中，MRW 和 fGn 的 `p_MRW` 很接近，甚至某些压缩表里 fGn 略高。

所以不能写：

```text
CMIN-SR successfully separates raw fGn and MRW.
```

只能写：

```text
The analytic interpretation layer separates controlled spectral geometries,
while raw finite-sample spectra expose the empirical zeta-estimation bottleneck.
```

---

### 4.4 real-data sanity check 不能写成 real-world validation

Fama-French 或市场因子 sanity check 只能说明：

```text
diagnostics show original-vs-shuffled differences consistent with temporal
multiscale dependence.
```

不能说明：

```text
real market data are MRW；
CMIN-SR validates real multifractality；
diagnostic proves mechanism。
```

建议 real-data 部分放 appendix 或 exploratory sanity check。

---

## 5. 应该保留的 claim

以下 claim 证据相对强，可以保留：

1. CMIN-SR 是 validity-aware spectral diagnostic framework。
2. `lambda2_proj` 是 projection coordinate，不是 mechanism proof。
3. Analytic spectrum-space 中 linear / boundary / curved / unstable geometry 可学习、可分。
4. Tested short-window grid 下 `lambda2` recovery weak。
5. fGn / Gaussian / heavy-tail / GARCH / regime controls 说明单一 `p_MRW` 不足，必须联合 residuals 和 instability 解释。
6. real-data sanity check 只能作为 exploratory evidence。

---

## 6. 必须软化的 claim

| 原始或潜在说法 | 建议改法 |
|---|---|
| finite-sample identifiability result | empirical finite-sample limitation under tested settings |
| calibrated probability | calibrated diagnostic / compatibility score |
| true MRW parameter | known simulation parameter |
| true MRW curvature | generated-MRW curvature |
| validates analytic spectral geometry | checks / tests analytic spectral geometry |
| proves MRW mechanism | organizes MRW-compatible evidence |
| classical estimators fail | tested structure-function estimators show weak recovery |
| real-data validation | exploratory real-data sanity check |

---

## 7. 建议直接加入论文的关键句

### 7.1 关于结构函数对象

```text
The input is a standardized return or increment sequence x_t. Structure
functions are intended to act on aggregated increments of the associated path,
Delta_a X_t = sum_{i=0}^{a-1} x_{t+i}. Implementation variants differ in
overlapping versus non-overlapping aggregation and are treated as estimator-level
diagnostics rather than identical realizations of a single ideal estimator.
```

### 7.2 关于 projection

```text
We fit the monofractal and MRW families by least-squares coordinates on the
selected q-grid, followed by admissibility clipping of H and lambda2. The
resulting lambda2_proj is a projection coordinate, not a generative parameter
estimate.
```

### 7.3 关于 `f(alpha)`

```text
For empirical spectra, alpha(q) and f(alpha(q)) are finite-grid Legendre
approximations over the tested positive q-grid, not a full singularity-spectrum
reconstruction.
```

### 7.4 关于 MRW simulation

```text
The synthetic MRW generator targets the parabolic MRW spectrum as a reference
family in the asymptotic/model-design sense, but finite length, integral-scale
truncation, normalization, and estimator noise mean that realized samples need
not exhibit the analytic curvature reliably.
```

### 7.5 关于 calibrated scores

```text
Here calibrated denotes model-trained diagnostic normalization, not verified
statistical probability calibration.
```

### 7.6 关于最终定位

```text
CMIN-SR does not prove MRW mechanisms from short stochastic samples. It organizes
finite-sample multifractal evidence by separating empirical spectrum estimation
from spectral-geometry interpretation.
```

---

## 8. 建议后续实际修改优先级

### Priority 1: 投稿前必须处理

1. 统一或解释 structure-function estimator 的数学对象。
2. 在 projection 公式附近说明实现是 least-squares coordinate fit + clipping。
3. 第一次出现 calibrated diagnostic score 时声明不是统计概率校准。
4. 所有 “true MRW” 改成 “known simulation parameter / generated MRW”。
5. 所有 finite-sample identifiability 写成 empirical/tested limitation。

### Priority 2: 强烈建议处理

1. 在 Methods/Experiments 加 process-role table。
2. 对 `f(alpha)` 加 finite-grid / positive-q limitation。
3. 将 real-data sanity check 放 appendix 或弱化。
4. 把 “physics-constrained” 降温成 “spectral-geometry-constrained”。

### Priority 3: 有时间再处理

1. 统一 `robust_zeta_estimator.py` 和论文结构函数口径。
2. 增加 wavelet leader / WTMM / full MFDFA baseline。
3. 增加 reliability-style calibration diagnostics，如果还想继续使用 calibrated probability 语言。

---

## 9. 总结

理论不是站不住，而是需要把边界说清楚。

最危险的不是 MRW 公式本身，而是：

```text
1. structure function 的实现口径不完全统一；
2. calibrated score 容易被误解成统计概率；
3. finite-sample identifiability 容易被误解成数学定理；
4. lambda2_proj 容易被误解成真实生成参数；
5. p_curved 容易被误解成 MRW 凹曲率，而它本质上先是非线性幅度。
```

只要把这些边界写清楚，项目主线是合理的，并且更像一篇诚实的有限样本多重分形谱诊断论文，而不是一个过度宣称的神经网络恢复模型。
