"""CLI demo: runs the Avellaneda-Stoikov market maker against a naive fixed-spread
baseline on identical simulated order flow (same seed), and reports a side-by-side
comparison of P&L, risk, and execution metrics. Saves comparison plots to results/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hft_mm import metrics
from hft_mm.backtester import Backtester
from hft_mm.market_maker import AvellanedaStoikovMarketMaker, NaiveMarketMaker

SEED = 7
N_TICKS = 20_000
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_strategy(strategy, seed: int = SEED, n_ticks: int = N_TICKS):
    """Run one backtest and return (raw result dict, summary metrics dict)."""
    bt = Backtester(strategy, n_ticks=n_ticks, seed=seed)
    result = bt.run()
    summary = metrics.summarize_run(
        result["pnl_series"],
        result["inventory_series"],
        result["fills"],
        result["mid_series"],
        result["n_quotes"],
        result["spread_series"],
    )
    return result, summary


def print_summary_table(as_summary: dict, naive_summary: dict) -> None:
    rows = [
        ("total_pnl", "Total P&L ($)", "{:.2f}"),
        ("sharpe", "Sharpe Ratio", "{:.2f}"),
        ("sortino", "Sortino Ratio", "{:.2f}"),
        ("max_drawdown", "Max Drawdown ($)", "{:.2f}"),
        ("avg_abs_inventory", "Avg |Inventory|", "{:.1f}"),
        ("max_abs_inventory", "Max |Inventory|", "{:.1f}"),
        ("fill_rate", "Fill Rate", "{:.1%}"),
        ("avg_spread", "Avg Spread ($)", "{:.4f}"),
        ("adverse_selection", "Adverse Selection ($)", "{:.4f}"),
    ]
    print(f"{'Metric':<24}{'Avellaneda-Stoikov':>20}{'Naive Baseline':>20}")
    print("-" * 64)
    for key, label, fmt in rows:
        as_val = fmt.format(as_summary.get(key, 0.0))
        naive_val = fmt.format(naive_summary.get(key, 0.0))
        print(f"{label:<24}{as_val:>20}{naive_val:>20}")


def make_plots(as_result: dict, naive_result: dict, out_dir: str = RESULTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

    ax1.plot(as_result["pnl_series"], label="Avellaneda-Stoikov", linewidth=1.2)
    ax1.plot(naive_result["pnl_series"], label="Naive baseline", linewidth=1.2)
    ax1.set_title("Cumulative P&L")
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(as_result["inventory_series"], label="Avellaneda-Stoikov", alpha=0.8)
    ax2.plot(naive_result["inventory_series"], label="Naive baseline", alpha=0.8)
    ax2.set_title("Inventory Position")
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Shares")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.plot(as_result["mid_series"], color="black", linewidth=0.8)
    ax3.set_title("Simulated Mid Price")
    ax3.set_xlabel("Tick")
    ax3.set_ylabel("Price")
    ax3.grid(alpha=0.3)

    ax4.hist(as_result["spread_series"], bins=40, alpha=0.6, label="AS book spread")
    ax4.hist(naive_result["spread_series"], bins=40, alpha=0.6, label="Naive book spread")
    ax4.set_title("Realized Book Spread Distribution")
    ax4.set_xlabel("Spread ($)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    combined_path = os.path.join(out_dir, "backtest_comparison.png")
    fig.savefig(combined_path, dpi=120)
    plt.close(fig)
    return combined_path


def main() -> None:
    print(f"Running {N_TICKS}-tick backtest (seed={SEED})...\n")

    as_result, as_summary = run_strategy(AvellanedaStoikovMarketMaker())
    naive_result, naive_summary = run_strategy(NaiveMarketMaker())

    print_summary_table(as_summary, naive_summary)

    plot_path = make_plots(as_result, naive_result)
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
