"""
worldmap.py

Cartopy + matplotlib world map of rogue-wave risk.
Interactive viewer: zoom, time slider, collected vs projected.
"""

import os

import numpy as np
import pandas as pd

from .maprender import aggregate_places, flagged_places, time_bins


CARTOPY_HINT = (
    'Cartopy is required for the world map. Install with: pip install ".[pipeline]" '
    '(cartopy needs GEOS/PROJ; on macOS: brew install geos proj)'
)


def require_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise RuntimeError(CARTOPY_HINT) from exc
    return ccrs, cfeature


def bin_time_label(left, right):
    """Human datetime range for one slider bin."""
    start = pd.Timestamp(left).strftime('%Y-%m-%d %H:%M')
    end = pd.Timestamp(right).strftime('%Y-%m-%d %H:%M')
    return '{}  ->  {}'.format(start, end)


def forecast_split_index(places, bins):
    """First bin that is projected (holds any time-holdout / test row)."""
    if not bins:
        return 0
    if 'split' not in places.columns or not (places['split'] == 'test').any():
        return len(bins)
    first_test = pd.to_datetime(places.loc[places['split'] == 'test', 'wave_start_time']).min()
    for i, (left, right, _frame) in enumerate(bins):
        if left <= first_test < right or left >= first_test:
            return i
    return len(bins)


def collected_projected(frame):
    """Split a bin into collected (train) vs projected (test) tables."""
    if frame is None or frame.empty:
        empty = frame if frame is not None else pd.DataFrame()
        return empty, empty
    if 'split' not in frame.columns:
        return frame, frame.iloc[0:0]
    collected = frame.loc[frame['split'] != 'test']
    projected = frame.loc[frame['split'] == 'test']
    return collected, projected


def _point_style(places):
    dots = aggregate_places(places)
    if dots is None or dots.empty:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    lon = dots['meta_deploy_longitude'].astype(float).values
    lat = dots['meta_deploy_latitude'].astype(float).values
    if 'rogue_prob' in dots.columns:
        prob = dots['rogue_prob'].astype(float).clip(0.0, 1.0).values
    else:
        prob = np.full(len(dots), 0.5)
    return lon, lat, prob


def _sizes(prob):
    return 50.0 + 420.0 * np.asarray(prob, dtype=float)


def draw_world_map(places, title, path, show=False, fig=None):
    """Static PlateCarree snapshot (used by tests and --no-show fallbacks)."""
    ccrs, cfeature = require_cartopy()
    import matplotlib
    if not show and fig is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    collected, projected = collected_projected(places)
    proj = ccrs.PlateCarree()
    created = fig is None
    if created:
        fig = plt.figure(figsize=(13, 6.5))
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor='#b9d8e8')
    ax.add_feature(cfeature.LAND, facecolor='#efe6d5')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#333333')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#666666', alpha=0.45)
    ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35)

    lon, lat, prob = _point_style(collected)
    if len(lon):
        ax.scatter(
            lon, lat, c=prob, s=_sizes(prob), cmap='YlOrRd', vmin=0.0, vmax=1.0,
            marker='o', edgecolors='k', linewidths=0.45, transform=proj,
            zorder=5, label='Collected',
        )
    lon_p, lat_p, prob_p = _point_style(projected)
    if len(lon_p):
        ax.scatter(
            lon_p, lat_p, s=_sizes(prob_p), marker='^', facecolors='none',
            edgecolors='#c0392b', linewidths=1.4, transform=proj,
            zorder=6, label='Projected',
        )
        extra = ax.scatter(
            lon_p, lat_p, c=prob_p, s=_sizes(prob_p) * 0.35, cmap='YlOrRd',
            vmin=0.0, vmax=1.0, marker='^', edgecolors='none',
            transform=proj, zorder=6,
        )
        colorbar = fig.colorbar(extra, ax=ax, shrink=0.72, pad=0.04)
        colorbar.set_label('P(rogue)')
    elif len(lon):
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.04)
        colorbar.set_label('P(rogue)')

    ax.set_title(title)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    elif created:
        plt.close(fig)
    return path


class MapViewer(object):
    """One cartopy figure: zoom/pan, time slider, collected vs projected."""

    def __init__(self, places, run_id, png_path, bin_size='1h', show=True):
        require_cartopy()
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.lines import Line2D
        from matplotlib.widgets import Slider

        self.places = places
        self.run_id = run_id
        self.png_path = png_path
        self.bins = time_bins(places, bin_size=bin_size)
        if not self.bins:
            raise ValueError('No time windows to map')
        self.split_index = forecast_split_index(places, self.bins)
        self.index = 0
        self.proj = ccrs.PlateCarree()
        self.show = show
        self._updating = False

        n = len(self.bins)
        self.fig = plt.figure(figsize=(13, 7.4))
        self.ax = self.fig.add_axes([0.04, 0.24, 0.78, 0.68], projection=self.proj)
        self.ax.set_global()
        self.ax.add_feature(cfeature.OCEAN, facecolor='#b9d8e8')
        self.ax.add_feature(cfeature.LAND, facecolor='#efe6d5')
        self.ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#333333')
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#666666', alpha=0.45)
        self.ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.35)
        self.ax.legend(
            handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e07020',
                       markeredgecolor='k', markersize=10, label='Collected'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
                       markeredgecolor='#c0392b', markersize=10, label='Projected'),
            ],
            loc='lower left', framealpha=0.9,
        )
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.65, pad=0.02)
        cbar.set_label('P(rogue)')

        self.sc_collected = self.ax.scatter(
            [], [], s=[], c=[], cmap='YlOrRd', vmin=0, vmax=1, marker='o',
            edgecolors='k', linewidths=0.45, transform=self.proj, zorder=5,
        )
        self.sc_projected = self.ax.scatter(
            [], [], s=[], marker='^', facecolors='none', edgecolors='#c0392b',
            linewidths=1.5, transform=self.proj, zorder=6,
        )
        self.sc_projected_fill = self.ax.scatter(
            [], [], s=[], c=[], cmap='YlOrRd', vmin=0, vmax=1, marker='^',
            edgecolors='none', transform=self.proj, zorder=6,
        )

        ax_slider = self.fig.add_axes([0.08, 0.08, 0.70, 0.045])
        ax_slider.patch.set_alpha(0)
        vmax = max(n - 1, 1)
        split_x = min(self.split_index, vmax)
        ax_slider.axvspan(0, split_x, facecolor='#5b9bd5', alpha=0.55, zorder=0)
        ax_slider.axvspan(split_x, vmax, facecolor='#ed7d31', alpha=0.55, zorder=0)
        ax_slider.set_yticks([])
        self.slider = Slider(
            ax_slider, '', 0, vmax, valinit=0, valstep=1, color='#5b9bd5',
        )
        self.slider.on_changed(self._on_slider)

        self.fig.text(0.08, 0.035, 'Collected', color='#1f4e79', fontsize=9)
        if 0 < self.split_index < n:
            frac = float(self.split_index) / float(vmax)
            self.fig.text(
                0.08 + 0.70 * frac, 0.035, 'Now / forecast start',
                color='#7b3f00', fontsize=9, ha='center',
            )
        self.fig.text(0.78, 0.035, 'Projected', color='#c45911', fontsize=9, ha='right')
        self.time_text = self.fig.text(
            0.43, 0.145, '', ha='center', va='center', fontsize=13, fontweight='bold',
        )
        self.title_text = self.ax.set_title('')

        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.set_index(0, save=False)

    def zoom_at(self, lon, lat, scale):
        """Zoom map extent about (lon, lat). scale < 1 zooms in."""
        x0, x1, y0, y1 = self.ax.get_extent(crs=self.proj)
        width = (x1 - x0) * float(scale)
        height = (y1 - y0) * float(scale)
        self.ax.set_extent(
            [lon - width / 2.0, lon + width / 2.0,
             lat - height / 2.0, lat + height / 2.0],
            crs=self.proj,
        )
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        lon, lat = event.xdata, event.ydata
        if lon is None or lat is None:
            return
        scale = 0.8 if event.button == 'up' else 1.25
        self.zoom_at(lon, lat, scale)

    def _on_slider(self, value):
        if self._updating:
            return
        self.set_index(int(round(value)), save=True)

    def set_index(self, index, save=True):
        n = len(self.bins)
        self.index = max(0, min(n - 1, int(index)))
        left, right, frame = self.bins[self.index]
        collected, projected = collected_projected(frame)
        self._set_scatter(self.sc_collected, None, collected, filled=True)
        self._set_scatter(self.sc_projected, self.sc_projected_fill, projected, filled=False)

        label = bin_time_label(left, right)
        side = 'Projected' if self.index >= self.split_index else 'Collected'
        self.time_text.set_text('{}   ({})'.format(label, side))
        self.ax.set_title('{}   |   {}'.format(self.run_id, label))
        if hasattr(self, 'slider') and abs(self.slider.val - self.index) > 0.1:
            self._updating = True
            try:
                self.slider.set_val(self.index)
            finally:
                self._updating = False
        color = '#ed7d31' if self.index >= self.split_index else '#5b9bd5'
        try:
            self.slider.poly.set_facecolor(color)
        except Exception:
            pass
        self.fig.canvas.draw_idle()
        if save:
            self.save()
        return self.index

    def _set_scatter(self, outline, fill, frame, filled):
        lon, lat, prob = _point_style(frame)
        offsets = np.column_stack([lon, lat]) if len(lon) else np.zeros((0, 2))
        sizes = _sizes(prob) if len(lon) else np.zeros(0)
        outline.set_offsets(offsets)
        outline.set_sizes(sizes)
        if filled:
            outline.set_array(prob if len(lon) else np.zeros(0))
        if fill is not None:
            fill.set_offsets(offsets)
            fill.set_sizes(sizes * 0.4 if len(sizes) else sizes)
            fill.set_array(prob if len(lon) else np.zeros(0))

    def save(self):
        directory = os.path.dirname(self.png_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.fig.savefig(self.png_path, dpi=130, bbox_inches='tight')
        return self.png_path


def interactive_world_map(predictions, run_id, png_path, show=True, bin_size='1h',
                          min_prob=0.0, actual_only=False, autoplay=None):
    """Open (or save) the interactive world map. autoplay is seconds per bin or None."""
    places = flagged_places(
        predictions, min_prob=min_prob, actual_only=actual_only,
    )
    if places.empty:
        raise ValueError('No flagged places to map. Generate a report first.')

    viewer = MapViewer(places, run_id, png_path, bin_size=bin_size, show=show)
    import matplotlib.pyplot as plt

    if autoplay:
        if show:
            plt.ion()
            plt.show(block=False)
        try:
            for i in range(len(viewer.bins)):
                viewer.set_index(i, save=True)
                if show:
                    plt.pause(max(0.05, float(autoplay)))
        except KeyboardInterrupt:
            pass
        if show:
            plt.ioff()
            plt.show()
        else:
            plt.close(viewer.fig)
        return viewer

    if not show:
        last_collected = max(0, viewer.split_index - 1)
        viewer.set_index(last_collected, save=True)
        plt.close(viewer.fig)
        return viewer

    plt.show()
    return viewer


def watch_world_map(frames, delay, png_path, show=True):
    """Backward-compatible watch: prefer interactive autoplay when possible."""
    if not frames:
        return
    # Reconstruct a predictions-like table from frames if needed.
    import matplotlib.pyplot as plt
    require_cartopy()
    fig = plt.figure(figsize=(13, 6.5))
    if show:
        plt.ion()
        plt.show(block=False)
    try:
        for frame in frames:
            draw_world_map(
                frame['places'], frame['title'], png_path,
                show=False, fig=fig,
            )
            if show:
                fig.canvas.draw_idle()
                plt.pause(max(0.05, float(delay)))
    except KeyboardInterrupt:
        return
    finally:
        if not show:
            plt.close(fig)
