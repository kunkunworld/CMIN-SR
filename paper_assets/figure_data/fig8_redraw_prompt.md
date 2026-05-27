# Fig. 8 Redraw Prompt: Real-Data Sanity Check

Use the CSV files in this directory:

- `fig8_real_data_sanity_check.csv`
- `fig8_real_data_sanity_check_window_level.csv`
- `fig8_real_data_sanity_check_plot_stats.csv`

Recommended figure: a compact two-panel white-background journal figure.

Panel A: original minus fully shuffled gaps.

Panel B: original minus block-shuffled gaps.

Use factor series on the x-axis (`Mkt-RF`, `SMB`, `HML`, `Mom`) and mean gap on
the y-axis. Prefer two or three metrics only: projected lambda2 gap, MRW
diagnostic gap, and optionally f(alpha) width gap. Draw a zero reference line.
Use error bars from `sem_gap`.

Caption should frame the figure as an exploratory Fama-French real-data sanity
check. Do not claim MRW mechanism validation. Use wording such as:

`The gaps are small and are reported only as diagnostic responses to temporal
multiscale structure, not as evidence of an MRW generating mechanism.`
