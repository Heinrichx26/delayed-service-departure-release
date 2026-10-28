"""Render report precision from saved results; no solver calls."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/revision_20261028/substantive_strengthening'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/figures')
    OUTPUT = parser.parse_args().output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    precision = json.loads((OUT / 'interval_precision_v2/panel.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9.5,
        'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False})
    groups = {(r['response_width'], r['method']): r for r in precision['summary'] if r['radius'] == 4}
    widths = [1, 2, 4]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.65))
    for method, style, color, marker in [('all_available', '-', '#0072B2', 'o'),
                                        ('selective_available', '--', '#D55E00', 's')]:
        values = [groups[w, method]['mean_queries'] for w in widths]
        axes[0].plot(widths, values, style, color=color, marker=marker, markersize=4.5,
                     linewidth=1.9,
                     label='All available' if method == 'all_available' else 'Selective')
    for method, style, color, marker, label in [
        ('past_only', ':', '#009E73', None, 'Past observations'),
        ('training_static_menu', '--', '#D55E00', 's', 'Fixed menu'),
        ('selective_available', '-', '#0072B2', 'o', 'All available / selective')]:
        axes[1].plot(widths, [groups[w, method]['mean_waiting_minutes_per_aircraft'] for w in widths],
                     style, color=color, marker=marker, markersize=4.5, linewidth=1.9,
                     label=label)
    axes[0].set_ylabel('Mean requests per window')
    axes[1].set_ylabel('Post-release time (min/departure)')
    for index, ax in enumerate(axes):
        ax.set_xlabel('Report width (possible counts)')
        ax.set_xticks(widths)
        ax.grid(axis='y', color='#D9D9D9', linewidth=.55)
        ax.set_title('(a) Request count' if index == 0 else '(b) Time after gate release')
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.subplots_adjust(left=.105, right=.985, bottom=.35, top=.89, wspace=.40)
    legends = []
    for ax, handles, labels in [(axes[0], handles0, labels0),
                                (axes[1], handles1, labels1)]:
        position = ax.get_position()
        legends.append(fig.legend(handles, labels, frameon=False, loc='upper center',
            bbox_to_anchor=((position.x0 + position.x1) / 2, .18),
            ncol=1, fontsize=8, handlelength=2.1, borderaxespad=0,
            labelspacing=.45))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, legend in zip(axes, legends):
        label_box = ax.xaxis.label.get_window_extent(renderer)
        legend_box = legend.get_window_extent(renderer)
        assert label_box.y0 - legend_box.y1 >= 6 * fig.dpi / 72, \
            'The x-axis label and legend require at least six points of clearance.'
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUTPUT / ('confirmation_precision.' + extension),
                    dpi=190, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)

if __name__ == '__main__':
    main()
