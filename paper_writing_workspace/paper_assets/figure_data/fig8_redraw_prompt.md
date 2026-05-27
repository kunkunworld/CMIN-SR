# Fig. 8 Redraw Prompt: Real-Data Sanity Check

## Purpose

Redraw the exploratory Fama-French real-data sanity check in a cleaner journal
style. The figure should show that the CMIN-SR diagnostic pipeline can be run on
empirical factor-return series and that original windows show small differences
from shuffled surrogates. It must not imply validation of an MRW mechanism.

## Data files

Use these CSV files:

- Summary long format:
  `paper_assets/figure_data/fig8_real_data_sanity_check.csv`
- Window-level values:
  `paper_assets/figure_data/fig8_real_data_sanity_check_window_level.csv`
- Mean and uncertainty statistics:
  `paper_assets/figure_data/fig8_real_data_sanity_check_plot_stats.csv`

The same files are mirrored under the repository-level
`paper_assets/figure_data/` directory.

## Recommended main figure

Create a compact two-panel figure.

### Panel A: original vs fully shuffled gaps

- x-axis: factor series (`Mkt-RF`, `SMB`, `HML`, `Mom`)
- y-axis: mean gap, defined as original minus shuffled
- show two or three metrics only:
  - projected lambda2 gap (`lambda2_gap`)
  - MRW diagnostic gap (`p_MRW_gap`)
  - optional spectrum-width gap (`f_alpha_width_gap`)
- use grouped points or narrow bars with error bars from `sem_gap`
- include a horizontal zero line
- title: `Original minus fully shuffled`

### Panel B: original vs block-shuffled gaps

- same x-axis and y-axis style
- same metrics
- title: `Original minus block-shuffled`
- emphasize that block shuffling largely reduces or reverses the gaps

## Alternative if space is tight

Use only one panel:

- x-axis: factor series
- y-axis: mean gap
- metric: `p_MRW_gap`
- two colors: fully shuffled and block shuffled
- error bars: `sem_gap`
- horizontal zero line

This is the safest version for the main text because it is easy to read and
does not overstate the real-data result.

## Visual style

- White background.
- No dark theme.
- Use restrained colors:
  - fully shuffled: muted blue (`#3B74D7`)
  - block shuffled: muted orange or gray (`#D59B37` or `#7A8795`)
  - zero line: light gray dashed line
- Keep labels large enough for a two-column PDF.
- Avoid dense boxplots in the main text; if distributions are needed, put
  window-level boxplots in the appendix.
- Use mathematical notation only where helpful:
  - `gap = original - surrogate`
  - `lambda2_proj` or `projected lambda^2`, not true lambda2.

## Caption suggestion

Exploratory real-data sanity check on Fama-French factor returns. Each point or
bar shows the mean rolling-window gap between the original factor series and a
surrogate series. Fully shuffled surrogates remove temporal ordering, whereas
block-shuffled surrogates preserve more local dependence. The gaps are small and
are reported only as diagnostic responses to temporal multiscale structure, not
as evidence of an MRW generating mechanism.

## Important wording constraints

Do not label the y-axis as true MRW strength. Use `projected lambda2` or
`diagnostic score`.

Do not claim real-world validation. The correct phrase is:

`exploratory real-data sanity check`.

Do not claim that Fama-French factors are MRW. The figure only shows whether the
diagnostic changes when temporal order is disrupted.

## Fields

`fig8_real_data_sanity_check.csv`

- `factor`: short factor name.
- `source_name`: original data source name.
- `n_windows`: number of rolling windows.
- `metric`: metric identifier.
- `metric_label`: human-readable metric name.
- `surrogate_type`: `fully_shuffled` or `block_shuffled`.
- `mean_gap_original_minus_surrogate`: mean original-minus-surrogate gap.

`fig8_real_data_sanity_check_plot_stats.csv`

- `factor`
- `metric`
- `metric_label`
- `surrogate_type`
- `n_windows`
- `mean_gap`
- `sem_gap`
- `median_gap`
- `q25_gap`
- `q75_gap`

`fig8_real_data_sanity_check_window_level.csv`

- rolling-window original, shuffled, block-shuffled values and gaps for
  projected lambda2, p_MRW proxy, f(alpha) width, log-volatility covariance
  slope, and H.
