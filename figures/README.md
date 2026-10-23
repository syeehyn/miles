# Routing replay forcing figures

Context: [radixark/miles#<RFC>](https://github.com/radixark/miles/issues/<RFC>).

- `fig-1-what-happens.png`: one multi-turn session across a weight sync, without and with forcing; which routes the trainer replays versus which routes sampled each token. Made by `make_design_figs.py`.
- `fig-2-how-it-works.png`: the request path through the Miles session server, the SGLang router and the SGLang engine, with the held-routes store and the fail-closed checks. Made by `make_design_figs.py`.
- `fig-3-off-vs-on.png`: forcing OFF (default) and ON, per step over 100 steps: log-prob gap and its per-step ON − OFF difference with the mean and 95% interval, KL, TIS clip fraction, retracted, forced and refused turns, dropped groups, weight-sync time, reward. Made by `plot_results.py`.
- `fig-4-on-minus-off.png`: the paired ON − OFF difference per step, with 10-step trailing means and, in (a), the running mean and its 95% interval. Made by `plot_diff.py`.

Figures 1 and 2 hold no experiment data. Figures 3 and 4 read the two runs' Miles logs with `plot_results.py` and `plot_diff.py` from `examples/experimental/routing_replay_forcing_ab/` on the test-only A/B branch, `routing-replay-forcing-ab`. The scripts need numpy and matplotlib.

```bash
python make_design_figs.py
# From the root of the A/B branch, with each arm's log saved by `run_ab.py train ... 2>&1 | tee off.log`:
python examples/experimental/routing_replay_forcing_ab/plot_results.py --source local --off off.log --on on.log \
    --title "Routing replay forcing, OFF vs ON (Qwen3.8-Flash-Next, 100 steps)"
python examples/experimental/routing_replay_forcing_ab/plot_diff.py --off off.log --on on.log \
    --subtitle Qwen3.8-Flash-Next
```

`make_design_figs.py` writes next to itself; the other two write to `ab-results/`.
