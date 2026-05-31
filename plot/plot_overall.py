import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Visual Style Settings ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

TITLE_FONTSIZE = 18
FONTSIZE = 14
X_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 18
TICKLABEL_FONTSIZE = 12
LINEWIDTH = 2
MARKERSIZE = 10
FIGSIZE_X = 20  # 宽度保持不变
FIGSIZE_Y = 20  # 高度加倍以适应6行
BOTTOM = 0.08  # 减小底部边距，让图例更靠近图形
TOP = 0.95  # 稍微调整以容纳更多行
LEFT = 0.06
RIGHT = 0.98

# Markers and Colors
marker_list = ["s", "^", "o", "x", "v"]
color_list = ["#5372ab", "#936bb9", "#6aa56e", "#c9b97d", "#b75555"]

# --- Configuration ---
DATASETS = ['cora', 'pubmed', 'facebook', 'lastfm']
MECHANISMS = ['mbm', 'pm', 'sr']
FIXED_MODEL = 'sage'
FIXED_Y_EPS = 1.0
FIXED_x_eps = 1.0
FIXED_INFLUENCE = 0.1

# X ticks for both plots
X_TICKS_EPS_X = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
X_TICKS_EPS_Y = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

DATA_DIR = './final_result/data/overall/'

# Attack Configurations (shared by both plots)
configs = [
    {'name': 'No Attack', 'use_baseline': True},
    {'name': 'DH', 'use_baseline': False, 'target_selection': 'random',
     'poison_strategy': 'random-index-based', 'label_popularity': 'random', 'edge_strategy': '0-hop'},
    {'name': 'SIGMA-DI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '0-hop'},
    {'name': 'SIGMA-LNI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '1-hop-1-to-1'},
    {'name': 'SIGMA-GNI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '1-hop-x-to-1'}
]


def load_eps_x_data():
    """Load data for epsilon_x plots"""
    data_map = {}
    for d_name in DATASETS:
        path = os.path.join(DATA_DIR, f'{d_name}.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['dataset'] = df['dataset'].str.lower()
                df['mechanism'] = df['mechanism'].str.lower()
                df['model'] = df['model'].str.lower()
                data_map[d_name] = df
                print(f"Loaded eps_x {d_name}: {df.shape}")
            except Exception as e:
                print(f"Error loading eps_x {d_name}: {e}")
                data_map[d_name] = pd.DataFrame()
        else:
            print(f"Warning: eps_x File not found {path}")
            data_map[d_name] = pd.DataFrame()
    return data_map


def load_eps_y_data():
    """Load data for epsilon_y plots"""
    data_map = {}
    for d_name in DATASETS:
        path = os.path.join(DATA_DIR, f'{d_name}_y.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['dataset'] = df['dataset'].str.lower()
                df['mechanism'] = df['mechanism'].str.lower()
                df['model'] = df['model'].str.lower()
                data_map[d_name] = df
                print(f"Loaded eps_y {d_name}: {df.shape}")
            except Exception as e:
                print(f"Error loading eps_y {d_name}: {e}")
                data_map[d_name] = pd.DataFrame()
        else:
            print(f"Warning: eps_y File not found {path}")
            data_map[d_name] = pd.DataFrame()
    return data_map


def plot_eps_x_lines(ax, df_subset):
    """Plot lines for epsilon_x (original first plot)"""
    if df_subset.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    for i, config in enumerate(configs):
        if config['use_baseline']:
            df_grouped = df_subset.groupby('x_eps').first().reset_index()
            df_sorted = df_grouped.sort_values('x_eps')
            x_vals = df_sorted['x_eps']
            y_vals = df_sorted['baseline_acc']
            err_vals = df_sorted.get('baseline_ci', [0] * len(y_vals))
        else:
            query_parts = []
            for k, v in config.items():
                if k not in ['name', 'use_baseline']:
                    if isinstance(v, str):
                        query_parts.append(f"{k} == '{v}'")
                    else:
                        query_parts.append(f"{k} == {v}")
            query = ' & '.join(query_parts)
            df_config = df_subset.query(query)

            if df_config.empty:
                continue

            df_sorted = df_config.sort_values('x_eps')
            x_vals = df_sorted['x_eps']
            y_vals = df_sorted['attack_acc']
            err_vals = df_sorted.get('attack_ci', [0] * len(y_vals))

        ax.errorbar(x_vals, y_vals, yerr=err_vals,
                    label=config['name'], color=color_list[i],
                    marker=marker_list[i], markersize=MARKERSIZE,
                    linewidth=LINEWIDTH, capsize=0,
                    markerfacecolor='none', markeredgewidth=2)


def plot_eps_y_lines(ax, df_subset, x_mapping):
    """Plot lines for epsilon_y (original second plot)"""
    if df_subset.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    for i, config in enumerate(configs):
        if config['use_baseline']:
            df_grouped = df_subset.groupby('y_eps').first().reset_index()
            df_sorted = df_grouped.sort_values('y_eps')
            x_vals_raw = df_sorted['y_eps']
            y_vals = df_sorted['baseline_acc']
            err_vals = df_sorted.get('baseline_ci', [0] * len(y_vals))
        else:
            query_parts = []
            for k, v in config.items():
                if k not in ['name', 'use_baseline']:
                    if isinstance(v, str):
                        query_parts.append(f"{k} == '{v}'")
                    else:
                        query_parts.append(f"{k} == {v}")
            query = ' & '.join(query_parts)
            df_config = df_subset.query(query)

            if df_config.empty:
                continue

            df_sorted = df_config.sort_values('y_eps')
            x_vals_raw = df_sorted['y_eps']
            y_vals = df_sorted['attack_acc']
            err_vals = df_sorted.get('attack_ci', [0] * len(y_vals))

        valid_indices = [idx for idx, x in enumerate(x_vals_raw) if x in x_mapping]
        if not valid_indices:
            continue

        x_plot = [x_mapping[x_vals_raw.iloc[idx]] for idx in valid_indices]
        y_plot = y_vals.iloc[valid_indices]
        err_plot = list(np.array(err_vals)[valid_indices])

        ax.errorbar(x_plot, y_plot, yerr=err_plot,
                    label=config['name'], color=color_list[i],
                    marker=marker_list[i], markersize=MARKERSIZE,
                    linewidth=LINEWIDTH, capsize=0,
                    markerfacecolor='none', markeredgewidth=2)


def main():
    # Load data for both plots
    data_map_eps_x = load_eps_x_data()
    data_map_eps_y = load_eps_y_data()

    # Create mapping for epsilon_y x-axis
    x_mapping_y = {val: i for i, val in enumerate(X_TICKS_EPS_Y)}

    # Create figure with 6 rows (3 for eps_x, 3 for eps_y) and 4 columns
    fig, axes = plt.subplots(6, 4, figsize=(FIGSIZE_X, FIGSIZE_Y),
                             sharey=False, sharex=False)

    # 调整布局：减小底部边距，增加上下子图之间的间距
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=0.25, wspace=0.1)

    # Plot epsilon_x rows (top 3 rows)
    print("\nPlotting epsilon_x plots (top 3 rows)...")
    for row_idx, mech in enumerate(MECHANISMS):
        for col_idx, d_name in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]

            # Set title for first row
            if row_idx == 0:
                letter = chr(97 + col_idx)
                formatted_name = d_name.title() if d_name != 'lastfm' else 'LastFM'
                if d_name == 'lastfm':
                    formatted_name = 'LastFM'
                else:
                    formatted_name = d_name.title()
                ax.set_title(f"({letter}) {formatted_name}", fontsize=TITLE_FONTSIZE, fontweight='medium')

            df = data_map_eps_x.get(d_name, pd.DataFrame())

            if not df.empty:
                mask = (
                        (df['model'] == FIXED_MODEL) &
                        (df['mechanism'] == mech) &
                        (df['influence_ratio'] == FIXED_INFLUENCE) &
                        (df['y_eps'] == FIXED_Y_EPS)
                )
                df_subset = df[mask]
                plot_eps_x_lines(ax, df_subset)

            # X-axis settings for eps_x
            ax.set_xticks(X_TICKS_EPS_X)
            ax.set_xlabel(r'$\epsilon_x$', fontsize=X_LABEL_FONTSIZE, fontweight='medium')

            # Grid and tick settings
            ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=1.0)
            ax.tick_params(axis='both', which='major', labelsize=TICKLABEL_FONTSIZE)

            # Y-axis labels
            if col_idx == 0:
                ax.set_ylabel(f"{mech.upper()}\nAccuracy", fontsize=FONTSIZE, fontweight='medium')
            else:
                ax.set_ylabel('')

    # Plot epsilon_y rows (bottom 3 rows)
    print("\nPlotting epsilon_y plots (bottom 3 rows)...")
    for row_idx_offset, mech in enumerate(MECHANISMS):
        row_idx = row_idx_offset + 3  # Offset by 3 for bottom section

        for col_idx, d_name in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]

            df = data_map_eps_y.get(d_name, pd.DataFrame())

            if not df.empty:
                mask = (
                        (df['model'] == FIXED_MODEL) &
                        (df['mechanism'] == mech) &
                        (df['influence_ratio'] == FIXED_INFLUENCE) &
                        (df['x_eps'] == FIXED_x_eps)
                )
                df_subset = df[mask]
                plot_eps_y_lines(ax, df_subset, x_mapping_y)

            # X-axis settings for eps_y
            ax.set_xticks(range(len(X_TICKS_EPS_Y)))
            ax.set_xticklabels([str(x) for x in X_TICKS_EPS_Y], fontweight='medium')
            ax.set_xlabel(r'$\epsilon_y$', fontsize=X_LABEL_FONTSIZE, fontweight='medium')

            # Grid and tick settings
            ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=1.0)
            ax.tick_params(axis='both', which='major', labelsize=TICKLABEL_FONTSIZE)

            # Y-axis labels
            if col_idx == 0:
                ax.set_ylabel(f"{mech.upper()}\nAccuracy", fontsize=FONTSIZE, fontweight='medium')
            else:
                ax.set_ylabel('')

    # Create a single legend at the bottom
    # Get handles and labels from first plot
    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        # 将图例位置向上调整，使其更靠近最后一行
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01),  # 向上调整，从0.01改为0.03
                   ncol=5, fontsize=LEGEND_FONTSIZE,
                   frameon=False,
                   shadow=False,
                   borderpad=1)
    else:
        print("Warning: No legend handles found.")

    # Save the combined figure
    output_dir = './final_result/figure'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'overall_sage.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nCombined figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()