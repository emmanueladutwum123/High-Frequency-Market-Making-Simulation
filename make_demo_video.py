"""Generate an animated replay of the AS vs. naive backtest as an MP4: price path,
inventory, and cumulative P&L building up tick-by-tick for both strategies.
Not part of the package/tests — a one-off artifact generator for a demo video.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from hft_mm.backtester import Backtester
from hft_mm.market_maker import AvellanedaStoikovMarketMaker, NaiveMarketMaker

SEED = 7
N_TICKS = 20_000
STRIDE = 20  # subsample for a smooth ~20s video instead of 20,000 raw frames
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "simulation_demo.mp4")

# Terminal-style palette shared with the artifact page wrapping this video.
INK = "#10161F"
PANEL = "#161D28"
TEXT = "#E7E4DC"
MUTED = "#6B7484"
GRID = "#232B38"
AMBER = "#D6A24C"   # Avellaneda-Stoikov
STEEL = "#5C86A6"   # naive baseline
MONO = ["Menlo", "DejaVu Sans Mono", "Consolas", "monospace"]


def run(strategy):
    bt = Backtester(strategy, n_ticks=N_TICKS, seed=SEED)
    return bt.run()


def main():
    print("Running backtests...")
    as_result = run(AvellanedaStoikovMarketMaker())
    naive_result = run(NaiveMarketMaker())

    mid = as_result["mid_series"][::STRIDE]
    as_pnl = as_result["pnl_series"][::STRIDE]
    naive_pnl = naive_result["pnl_series"][::STRIDE]
    as_inv = as_result["inventory_series"][::STRIDE]
    naive_inv = naive_result["inventory_series"][::STRIDE]
    ticks = np.arange(len(mid)) * STRIDE

    n_frames = len(mid)
    print(f"{n_frames} frames")

    plt.rcParams["font.family"] = MONO
    plt.rcParams["text.color"] = TEXT
    plt.rcParams["axes.labelcolor"] = MUTED
    plt.rcParams["xtick.color"] = MUTED
    plt.rcParams["ytick.color"] = MUTED

    fig, (ax_price, ax_inv, ax_pnl) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(INK)

    fig.text(
        0.045, 0.965, "HIGH-FREQUENCY MARKET MAKING SIMULATION",
        fontsize=11, fontweight="bold", color=TEXT, family=MONO,
    )
    fig.text(
        0.045, 0.94, "Avellaneda-Stoikov vs. naive fixed-spread — identical order flow, seed 7",
        fontsize=8.5, color=MUTED, family=MONO,
    )

    axes = (ax_price, ax_inv, ax_pnl)
    labels = ("MID PRICE ($)", "INVENTORY (SHARES)", "CUMULATIVE P&L ($)")
    for ax, label in zip(axes, labels):
        ax.set_facecolor(PANEL)
        ax.set_ylabel(label, fontsize=8.5, family=MONO)
        ax.set_xlim(0, ticks[-1])
        ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(labelsize=8)

    ax_price.set_ylim(mid.min() - 0.2, mid.max() + 0.2)
    (price_line,) = ax_price.plot([], [], color=TEXT, linewidth=1.1)

    inv_lim = (
        max(as_inv.max(initial=0), naive_inv.max(initial=0), -as_inv.min(initial=0), -naive_inv.min(initial=0))
        + 20
    )
    ax_inv.set_ylim(-inv_lim, inv_lim)
    ax_inv.axhline(0, color=GRID, linewidth=0.8)
    (as_inv_line,) = ax_inv.plot([], [], color=AMBER, label="Avellaneda-Stoikov", linewidth=1.2)
    (naive_inv_line,) = ax_inv.plot([], [], color=STEEL, label="Naive baseline", linewidth=1.2)
    leg = ax_inv.legend(loc="upper left", fontsize=7.5, frameon=False)
    for text in leg.get_texts():
        text.set_color(MUTED)

    ax_pnl.set_xlabel("TICK", fontsize=8.5, family=MONO)
    pnl_max = max(as_pnl.max(initial=0), naive_pnl.max(initial=0))
    pnl_min = min(as_pnl.min(initial=0), naive_pnl.min(initial=0))
    ax_pnl.set_ylim(pnl_min - 50, pnl_max + 50)
    (as_pnl_line,) = ax_pnl.plot([], [], color=AMBER, label="Avellaneda-Stoikov", linewidth=1.4)
    (naive_pnl_line,) = ax_pnl.plot([], [], color=STEEL, label="Naive baseline", linewidth=1.4)
    leg2 = ax_pnl.legend(loc="upper left", fontsize=7.5, frameon=False)
    for text in leg2.get_texts():
        text.set_color(MUTED)

    stats_text = fig.text(
        0.72, 0.965, "", fontsize=8.5, family=MONO, va="top", ha="left", color=TEXT
    )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.90])

    def update(frame):
        x = ticks[: frame + 1]
        price_line.set_data(x, mid[: frame + 1])
        as_inv_line.set_data(x, as_inv[: frame + 1])
        naive_inv_line.set_data(x, naive_inv[: frame + 1])
        as_pnl_line.set_data(x, as_pnl[: frame + 1])
        naive_pnl_line.set_data(x, naive_pnl[: frame + 1])

        stats_text.set_text(
            f"tick {ticks[frame]:>6d}\n"
            f"AS    {as_pnl[frame]:>+8.1f}\n"
            f"naive {naive_pnl[frame]:>+8.1f}"
        )
        return price_line, as_inv_line, naive_inv_line, as_pnl_line, naive_pnl_line, stats_text

    # hold the final frame for a couple seconds so the ending result is readable
    hold_frames = 60
    frame_indices = list(range(n_frames)) + [n_frames - 1] * hold_frames

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, blit=False, interval=1000 / 30)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Rendering to {OUT_PATH} ...")
    anim.save(
        OUT_PATH,
        writer=animation.FFMpegWriter(fps=30, bitrate=2400),
        savefig_kwargs={"facecolor": INK},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
