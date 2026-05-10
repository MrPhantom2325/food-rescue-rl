"""
Visualization layer for the food rescue environment.

The FrameRenderer class produces a single matplotlib Figure showing the
current state of an env: grid, donors, shelters, vehicles, targets, and
running metrics. Used both for static plots (in the report) and as the
per-frame builder for the animator (Step 11).

Color convention (kept consistent across the project):
- Donors:   green   (saturation = pending quantity)
- Shelters: orange  (saturation = current unmet demand)
- Vehicles: blue    (filled = loaded, hollow = empty)
- Background: dark slate
- Text: light slate

Usage:
    from sim.render import FrameRenderer
    renderer = FrameRenderer(env)
    fig = renderer.render(reward=12.5, step_info=info, total_reward=145.2)
    fig.savefig('frame.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from sim.environment import FoodRescueEnv


# -----------------------------
# Palette
# -----------------------------

@dataclass(frozen=True)
class Palette:
    """Centralized color choices for consistent visuals across the project."""
    background: str = "#0f172a"   # slate-900 — dark background
    grid: str = "#1e293b"          # slate-800 — subtle grid lines
    panel_bg: str = "#1e293b"      # info panel background
    text: str = "#e2e8f0"          # slate-200 — main text
    text_muted: str = "#94a3b8"    # slate-400 — secondary text
    donor: str = "#22c55e"         # green-500 — donors
    donor_dark: str = "#14532d"    # green-900 — empty donors
    shelter: str = "#f97316"       # orange-500 — shelters
    shelter_dark: str = "#7c2d12"  # orange-900 — sated shelters
    vehicle_loaded: str = "#3b82f6"  # blue-500 — loaded vehicles
    vehicle_empty: str = "#1e3a8a"   # blue-900 — empty vehicles
    target_line: str = "#64748b"     # slate-500 — vehicle target line
    priority: str = "#fbbf24"        # amber-400 — priority shelter halo


PALETTE = Palette()


# -----------------------------
# FrameRenderer
# -----------------------------

class FrameRenderer:
    """
    Build a matplotlib figure representing the current env state.

    The renderer is constructed once per episode (or per video) and reused
    across timesteps for efficiency. Each call to render() produces a
    fresh Figure.

    Parameters
    ----------
    env : FoodRescueEnv
        Must have been reset() at least once before render().
    figsize : tuple[float, float]
        Figure size in inches. Default 12×7 produces 1440×840 at 120dpi.
    show_grid : bool
        Whether to draw gridlines.
    show_legend : bool
        Whether to show the legend (donor/shelter/vehicle markers).
    """

    def __init__(
        self,
        env: FoodRescueEnv,
        figsize: tuple[float, float] = (12.0, 7.0),
        show_grid: bool = True,
        show_legend: bool = True,
    ):
        self.env = env
        self.figsize = figsize
        self.show_grid = show_grid
        self.show_legend = show_legend

    def render(
        self,
        reward: Optional[float] = None,
        total_reward: Optional[float] = None,
        step_info: Optional[dict] = None,
        title_extra: Optional[str] = None,
    ) -> Figure:
        """
        Produce a Figure representing the current env state.

        Parameters
        ----------
        reward : float, optional
            The reward from the most recent step (for the info panel).
        total_reward : float, optional
            Cumulative reward so far in the episode (for the info panel).
        step_info : dict, optional
            The info dict returned by env.step(). Used to surface per-step
            details like deliveries and spoilage.
        title_extra : str, optional
            Extra text appended to the figure title (e.g., agent name).

        Returns
        -------
        Figure
            A matplotlib figure ready to be saved or shown. The caller
            is responsible for closing it (plt.close(fig)) to free memory.
        """
        if self.env.scenario is None:
            raise RuntimeError("Cannot render env that hasn't been reset()")

        fig, (ax_grid, ax_info) = plt.subplots(
            1, 2, figsize=self.figsize,
            gridspec_kw={"width_ratios": [3, 1]},
        )
        fig.patch.set_facecolor(PALETTE.background)

        self._draw_grid(ax_grid)
        self._draw_donors(ax_grid)
        self._draw_shelters(ax_grid)
        self._draw_vehicles(ax_grid)
        self._draw_target_lines(ax_grid)
        if self.show_legend:
            self._draw_legend(ax_grid)
        self._draw_title(ax_grid, title_extra)

        self._draw_info_panel(ax_info, reward, total_reward, step_info)

        plt.tight_layout()
        return fig

    # -----------------------------
    # Grid
    # -----------------------------

    def _draw_grid(self, ax) -> None:
        gs = self.env.grid_size
        ax.set_facecolor(PALETTE.background)
        ax.set_xlim(-0.5, gs - 0.5)
        ax.set_ylim(-0.5, gs - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # so (0,0) is top-left like a city map

        if self.show_grid:
            for i in range(gs + 1):
                ax.axhline(i - 0.5, color=PALETTE.grid, linewidth=0.5, zorder=0)
                ax.axvline(i - 0.5, color=PALETTE.grid, linewidth=0.5, zorder=0)

        ax.set_xticks(range(gs))
        ax.set_yticks(range(gs))
        ax.tick_params(colors=PALETTE.text_muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(PALETTE.grid)

    # -----------------------------
    # Donors
    # -----------------------------

    def _draw_donors(self, ax) -> None:
        for d in self.env.scenario.donors:
            x, y = d.location
            qty = d.total_pending_quantity()
            min_sl = d.min_pending_shelf_life()

            # Marker size scales with pending quantity (clamped)
            size = 200 + min(qty * 25, 600)
            color = PALETTE.donor if qty > 0 else PALETTE.donor_dark

            # Halo if any batch is near spoilage
            if qty > 0 and min_sl <= 15:
                ax.scatter(
                    [x], [y], s=size + 350, c="none",
                    edgecolors="#fbbf24", linewidths=2.0,
                    zorder=2,
                )

            ax.scatter(
                [x], [y], s=size, c=color,
                edgecolors=PALETTE.text, linewidths=1.0,
                marker="s", zorder=3,
            )

            # Label below the marker
            label = f"{d.donor_id}\n{qty:.0f}u"
            if qty > 0:
                label += f" · {min_sl}t"
            ax.annotate(
                label, (x, y), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=7, color=PALETTE.text,
            )

    # -----------------------------
    # Shelters
    # -----------------------------

    def _draw_shelters(self, ax) -> None:
        for s in self.env.scenario.shelters:
            x, y = s.location
            demand = s.current_demand
            util = s.utilization()

            # Marker size scales with demand
            size = 200 + min(util * 600, 600)
            color = PALETTE.shelter if demand > 1 else PALETTE.shelter_dark

            # Priority halo for priority-1 shelters
            if s.priority == 1:
                ax.scatter(
                    [x], [y], s=size + 400, c="none",
                    edgecolors=PALETTE.priority, linewidths=1.5,
                    linestyle=":", zorder=2,
                )

            ax.scatter(
                [x], [y], s=size, c=color,
                edgecolors=PALETTE.text, linewidths=1.0,
                marker="o", zorder=3,
            )

            label = f"{s.shelter_id}\n{demand:.0f}u"
            ax.annotate(
                label, (x, y), xytext=(0, -22), textcoords="offset points",
                ha="center", fontsize=7, color=PALETTE.text,
            )

    # -----------------------------
    # Vehicles
    # -----------------------------

    def _draw_vehicles(self, ax) -> None:
        for v in self.env.vehicles:
            x, y = v.location
            loaded = v.current_load() > 0
            color = PALETTE.vehicle_loaded if loaded else PALETTE.vehicle_empty

            # Highlight the *currently acting* vehicle
            is_acting = v.vehicle_id == self.env.current_vehicle_idx
            edge = PALETTE.text if is_acting else PALETTE.text_muted
            edge_w = 2.5 if is_acting else 1.0

            ax.scatter(
                [x], [y], s=350, c=color,
                edgecolors=edge, linewidths=edge_w,
                marker="^", zorder=4,
            )

            label = f"V{v.vehicle_id}\n{v.current_load():.0f}/{v.capacity:.0f}"
            ax.annotate(
                label, (x, y), xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=7, color=PALETTE.text, fontweight="bold",
            )

    # -----------------------------
    # Vehicle target lines
    # -----------------------------

    def _draw_target_lines(self, ax) -> None:
        for v in self.env.vehicles:
            if v.target is None:
                continue
            x0, y0 = v.location
            x1, y1 = v.target
            ax.plot(
                [x0, x1], [y0, y1],
                color=PALETTE.target_line, linewidth=1.0,
                linestyle="--", alpha=0.6, zorder=1,
            )

    # -----------------------------
    # Title
    # -----------------------------

    def _draw_title(self, ax, title_extra: Optional[str]) -> None:
        scn_name = self.env.scenario.name
        step = self.env.current_step
        max_step = self.env.max_episode_steps
        bucket = self.env.scenario.city.time_bucket(min(step, max_step - 1))

        title = f"Food Rescue · {scn_name} · step {step}/{max_step} · {bucket}"
        if title_extra:
            title = f"{title} · {title_extra}"

        ax.set_title(title, color=PALETTE.text, fontsize=11, pad=10)

    # -----------------------------
    # Legend
    # -----------------------------

    def _draw_legend(self, ax) -> None:
        handles = [
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=PALETTE.donor, markeredgecolor=PALETTE.text,
                   markersize=10, label="Donor (green = food pending)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PALETTE.shelter, markeredgecolor=PALETTE.text,
                   markersize=10, label="Shelter (orange = unmet demand)"),
            Line2D([0], [0], marker="^", color="none",
                   markerfacecolor=PALETTE.vehicle_loaded, markeredgecolor=PALETTE.text,
                   markersize=10, label="Vehicle (filled = loaded)"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor=PALETTE.priority,
                   markersize=14, linestyle=":", label="Priority shelter (halo)"),
        ]
        legend = ax.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.06), ncol=2,
            fontsize=8, frameon=False,
            labelcolor=PALETTE.text_muted,
        )

    # -----------------------------
    # Info panel
    # -----------------------------

    def _draw_info_panel(
        self, ax,
        reward: Optional[float],
        total_reward: Optional[float],
        step_info: Optional[dict],
    ) -> None:
        ax.set_facecolor(PALETTE.panel_bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        em = self.env._episode_metrics
        lines = []

        # Top: scenario name + step
        lines.append(("RUN", "", "header"))
        lines.append(("scenario", self.env.scenario.name, "value"))
        lines.append(("step", f"{self.env.current_step}/{self.env.max_episode_steps}", "value"))

        # Reward
        lines.append(("", "", "spacer"))
        lines.append(("REWARD", "", "header"))
        if reward is not None:
            lines.append(("last step", f"{reward:+.2f}", "value"))
        if total_reward is not None:
            color_code = "good" if total_reward >= 0 else "bad"
            lines.append(("total", f"{total_reward:+.2f}", color_code))

        # Cumulative metrics
        lines.append(("", "", "spacer"))
        lines.append(("METRICS", "", "header"))
        lines.append(("delivered", f"{em['total_delivered_units']:.0f} units", "value"))
        lines.append(("spoiled", f"{em['total_spoiled_units']:.0f} units", "value"))
        lines.append(("distance", f"{em['total_distance']} cells", "value"))
        lines.append(("deliveries", f"{em['deliveries_count']}", "value"))

        # Last-step events (if any)
        if step_info:
            lines.append(("", "", "spacer"))
            lines.append(("LAST ACTION", "", "header"))
            kind = step_info.get("action_kind", "?")
            target = step_info.get("action_target_id", "")
            v_idx = step_info.get("vehicle_idx", "?")
            lines.append((f"V{v_idx}", f"{kind} {target or ''}".strip(), "value"))
            if step_info.get("delivered_units", 0) > 0:
                lines.append(("delivered", f"+{step_info['delivered_units']:.0f}u", "good"))
            if step_info.get("spoiled_units_donor", 0) + step_info.get("spoiled_units_vehicle", 0) > 0:
                spoiled_step = step_info["spoiled_units_donor"] + step_info["spoiled_units_vehicle"]
                lines.append(("spoiled", f"-{spoiled_step:.0f}u", "bad"))

        # Render lines
        y = 0.96
        for label, value, kind in lines:
            if kind == "spacer":
                y -= 0.025
                continue
            if kind == "header":
                ax.text(0.04, y, label, color=PALETTE.text_muted,
                        fontsize=8, fontweight="bold",
                        transform=ax.transAxes)
                y -= 0.04
                continue

            color = PALETTE.text
            if kind == "good":
                color = "#22c55e"
            elif kind == "bad":
                color = "#ef4444"

            ax.text(0.06, y, label, color=PALETTE.text_muted,
                    fontsize=9, transform=ax.transAxes)
            ax.text(0.96, y, value, color=color, fontsize=9,
                    ha="right", fontweight="bold", transform=ax.transAxes)
            y -= 0.04


# -----------------------------
# EpisodeAnimator
# -----------------------------

class EpisodeAnimator:
    """
    Run an episode and produce an animation.

    Two modes:
    - record(): runs the episode, captures every frame's state, then renders
      them all into a matplotlib FuncAnimation. Returns the animation.
    - save_mp4() / save_gif(): convenience wrappers that record() then export.

    The "capture state, render later" approach is intentional. Rendering during
    the env loop would mean recreating the figure every step, which is slow and
    creates frame-to-frame inconsistencies. Instead we snapshot env state into
    lightweight dicts during the loop, then render from snapshots after.

    Usage:
        from sim.render import EpisodeAnimator
        from agents.baseline import GreedyPolicy

        animator = EpisodeAnimator(env, GreedyPolicy())
        animator.save_mp4("experiments/videos/greedy_weekday.mp4", seed=42)
    """

    def __init__(
        self,
        env: FoodRescueEnv,
        policy,
        figsize: tuple[float, float] = (12.0, 7.0),
    ):
        self.env = env
        self.policy = policy
        self.figsize = figsize
        self._snapshots: list[dict] = []
        self._policy_name = getattr(policy, "name", "policy")

    def record(self, seed: Optional[int] = None, max_steps: Optional[int] = None) -> None:
        """
        Run a full episode (or up to max_steps) and capture per-step snapshots.

        After this returns, self._snapshots contains everything needed to render
        any frame retrospectively.
        """
        obs, _ = self.env.reset(seed=seed)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self._snapshots = []
        total_reward = 0.0
        step = 0
        max_steps = max_steps or self.env.max_episode_steps

        # Capture step-0 state (before any action)
        self._snapshots.append(self._capture_snapshot(reward=0.0, total_reward=0.0, info=None))

        while step < max_steps:
            action = self.policy.select_action(self.env, obs)
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            self._snapshots.append(
                self._capture_snapshot(reward=reward, total_reward=total_reward, info=info)
            )
            step += 1
            if term or trunc:
                break

    def _capture_snapshot(self, reward: float, total_reward: float, info: Optional[dict]) -> dict:
        """
        Snapshot the env state into a lightweight dict.

        We don't deep-copy the entire env (expensive); we copy just what the
        renderer reads. Crucially we copy *values*, not references, so later
        env changes don't corrupt the snapshot.
        """
        env = self.env
        return {
            "step": env.current_step,
            "current_vehicle_idx": env.current_vehicle_idx,
            "reward": reward,
            "total_reward": total_reward,
            "info": dict(info) if info else None,
            "donors": [
                {
                    "donor_id": d.donor_id,
                    "location": d.location,
                    "total_pending_quantity": d.total_pending_quantity(),
                    "min_pending_shelf_life": d.min_pending_shelf_life(),
                }
                for d in env.scenario.donors
            ],
            "shelters": [
                {
                    "shelter_id": s.shelter_id,
                    "location": s.location,
                    "current_demand": s.current_demand,
                    "capacity": s.capacity,
                    "priority": s.priority,
                    "utilization": s.utilization(),
                }
                for s in env.scenario.shelters
            ],
            "vehicles": [
                {
                    "vehicle_id": v.vehicle_id,
                    "location": v.location,
                    "current_load": v.current_load(),
                    "capacity": v.capacity,
                    "target": v.target,
                    "target_kind": v.target_kind,
                    "target_id": v.target_id,
                }
                for v in env.vehicles
            ],
            "episode_metrics": dict(env._episode_metrics),
        }

    def save_mp4(
        self,
        out_path: str,
        seed: Optional[int] = None,
        fps: int = 8,
        max_steps: Optional[int] = None,
    ) -> None:
        """Record an episode and save as MP4 (requires ffmpeg)."""
        import os
        from matplotlib.animation import FFMpegWriter

        if not self._snapshots:
            self.record(seed=seed, max_steps=max_steps)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        anim = self._build_animation(fps=fps)
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer, dpi=120,
                  savefig_kwargs={"facecolor": PALETTE.background})
        plt.close("all")

    def save_gif(
        self,
        out_path: str,
        seed: Optional[int] = None,
        fps: int = 8,
        max_steps: Optional[int] = None,
    ) -> None:
        """Record an episode and save as GIF (slower than MP4 but no ffmpeg needed)."""
        import os
        from matplotlib.animation import PillowWriter

        if not self._snapshots:
            self.record(seed=seed, max_steps=max_steps)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        anim = self._build_animation(fps=fps)
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=80,
                  savefig_kwargs={"facecolor": PALETTE.background})
        plt.close("all")

    def _build_animation(self, fps: int):
        """Create a FuncAnimation from the captured snapshots."""
        from matplotlib.animation import FuncAnimation

        fig, (ax_grid, ax_info) = plt.subplots(
            1, 2, figsize=self.figsize,
            gridspec_kw={"width_ratios": [3, 1]},
        )
        fig.patch.set_facecolor(PALETTE.background)

        def update(frame_idx: int):
            ax_grid.clear()
            ax_info.clear()
            snap = self._snapshots[frame_idx]
            self._render_snapshot(ax_grid, ax_info, snap)
            return []

        anim = FuncAnimation(
            fig, update,
            frames=len(self._snapshots),
            interval=1000 / fps,
            blit=False,
            repeat=False,
        )
        return anim

    def _render_snapshot(self, ax_grid, ax_info, snap: dict) -> None:
        """Render one snapshot into the given axes (animation-friendly version)."""
        env = self.env
        gs = env.grid_size

        # Grid
        ax_grid.set_facecolor(PALETTE.background)
        ax_grid.set_xlim(-0.5, gs - 0.5)
        ax_grid.set_ylim(-0.5, gs - 0.5)
        ax_grid.set_aspect("equal")
        ax_grid.invert_yaxis()
        for i in range(gs + 1):
            ax_grid.axhline(i - 0.5, color=PALETTE.grid, linewidth=0.5, zorder=0)
            ax_grid.axvline(i - 0.5, color=PALETTE.grid, linewidth=0.5, zorder=0)
        ax_grid.set_xticks(range(gs))
        ax_grid.set_yticks(range(gs))
        ax_grid.tick_params(colors=PALETTE.text_muted, labelsize=8)
        for spine in ax_grid.spines.values():
            spine.set_color(PALETTE.grid)

        # Donors
        for d in snap["donors"]:
            x, y = d["location"]
            qty = d["total_pending_quantity"]
            min_sl = d["min_pending_shelf_life"]
            size = 200 + min(qty * 25, 600)
            color = PALETTE.donor if qty > 0 else PALETTE.donor_dark
            if qty > 0 and min_sl <= 15:
                ax_grid.scatter([x], [y], s=size + 350, c="none",
                                edgecolors=PALETTE.priority, linewidths=2.0, zorder=2)
            ax_grid.scatter([x], [y], s=size, c=color,
                            edgecolors=PALETTE.text, linewidths=1.0,
                            marker="s", zorder=3)
            label = f"{d['donor_id']}\n{qty:.0f}u"
            if qty > 0:
                label += f" · {min_sl}t"
            ax_grid.annotate(label, (x, y), xytext=(0, -22),
                             textcoords="offset points",
                             ha="center", fontsize=7, color=PALETTE.text)

        # Shelters
        for s in snap["shelters"]:
            x, y = s["location"]
            demand = s["current_demand"]
            util = s["utilization"]
            size = 200 + min(util * 600, 600)
            color = PALETTE.shelter if demand > 1 else PALETTE.shelter_dark
            if s["priority"] == 1:
                ax_grid.scatter([x], [y], s=size + 400, c="none",
                                edgecolors=PALETTE.priority, linewidths=1.5,
                                linestyle=":", zorder=2)
            ax_grid.scatter([x], [y], s=size, c=color,
                            edgecolors=PALETTE.text, linewidths=1.0,
                            marker="o", zorder=3)
            ax_grid.annotate(f"{s['shelter_id']}\n{demand:.0f}u", (x, y),
                             xytext=(0, -22), textcoords="offset points",
                             ha="center", fontsize=7, color=PALETTE.text)

        # Vehicles
        for v in snap["vehicles"]:
            x, y = v["location"]
            loaded = v["current_load"] > 0
            color = PALETTE.vehicle_loaded if loaded else PALETTE.vehicle_empty
            is_acting = v["vehicle_id"] == snap["current_vehicle_idx"]
            edge = PALETTE.text if is_acting else PALETTE.text_muted
            edge_w = 2.5 if is_acting else 1.0
            ax_grid.scatter([x], [y], s=350, c=color,
                            edgecolors=edge, linewidths=edge_w,
                            marker="^", zorder=4)
            ax_grid.annotate(f"V{v['vehicle_id']}\n{v['current_load']:.0f}/{v['capacity']:.0f}",
                             (x, y), xytext=(0, 18), textcoords="offset points",
                             ha="center", fontsize=7, color=PALETTE.text, fontweight="bold")
            if v["target"] is not None:
                tx, ty = v["target"]
                ax_grid.plot([x, tx], [y, ty], color=PALETTE.target_line,
                             linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

        # Title
        bucket = env.scenario.city.time_bucket(min(snap["step"], env.max_episode_steps - 1))
        title = (f"Food Rescue · {env.scenario.name} · "
                 f"step {snap['step']}/{env.max_episode_steps} · {bucket} · {self._policy_name}")
        ax_grid.set_title(title, color=PALETTE.text, fontsize=11, pad=10)

        # Info panel
        ax_info.set_facecolor(PALETTE.panel_bg)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")

        em = snap["episode_metrics"]
        info = snap.get("info") or {}

        rows = [
            ("RUN", "", "header"),
            ("scenario", env.scenario.name, "value"),
            ("step", f"{snap['step']}/{env.max_episode_steps}", "value"),
            ("policy", self._policy_name, "value"),
            ("", "", "spacer"),
            ("REWARD", "", "header"),
            ("last step", f"{snap['reward']:+.2f}", "value"),
            ("total", f"{snap['total_reward']:+.2f}",
             "good" if snap["total_reward"] >= 0 else "bad"),
            ("", "", "spacer"),
            ("METRICS", "", "header"),
            ("delivered", f"{em['total_delivered_units']:.0f} units", "value"),
            ("spoiled", f"{em['total_spoiled_units']:.0f} units", "value"),
            ("distance", f"{em['total_distance']} cells", "value"),
            ("deliveries", f"{em['deliveries_count']}", "value"),
        ]

        if info:
            rows.extend([
                ("", "", "spacer"),
                ("LAST ACTION", "", "header"),
                (f"V{info.get('vehicle_idx', '?')}",
                 f"{info.get('action_kind', '?')} {info.get('action_target_id') or ''}".strip(),
                 "value"),
            ])
            if info.get("delivered_units", 0) > 0:
                rows.append(("delivered", f"+{info['delivered_units']:.0f}u", "good"))
            spoiled_step = (info.get("spoiled_units_donor", 0) +
                            info.get("spoiled_units_vehicle", 0))
            if spoiled_step > 0:
                rows.append(("spoiled", f"-{spoiled_step:.0f}u", "bad"))

        y = 0.96
        for label, value, kind in rows:
            if kind == "spacer":
                y -= 0.025
                continue
            if kind == "header":
                ax_info.text(0.04, y, label, color=PALETTE.text_muted,
                             fontsize=8, fontweight="bold", transform=ax_info.transAxes)
                y -= 0.04
                continue
            color = PALETTE.text
            if kind == "good":
                color = "#22c55e"
            elif kind == "bad":
                color = "#ef4444"
            ax_info.text(0.06, y, label, color=PALETTE.text_muted,
                         fontsize=9, transform=ax_info.transAxes)
            ax_info.text(0.96, y, value, color=color, fontsize=9,
                         ha="right", fontweight="bold", transform=ax_info.transAxes)
            y -= 0.04
