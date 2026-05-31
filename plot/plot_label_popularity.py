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
X_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 18
TICKLABEL_FONTSIZE = 12
LINEWIDTH = 2
FIGSIZE_X = 18  # 调整宽度以适应3列
FIGSIZE_Y = 12  # 调整高度以适应3行
BOTTOM = 0.12
TOP = 0.92
LEFT = 0.08
RIGHT = 0.98

# --- Configuration ---
# 1. Models (Rows)
MODELS = ['gcn', 'sage', 'gat']
# 2. Mechanisms (Columns)
MECHANISMS = ['mbm', 'pm', 'sr']

# Fixed Parameters
FIXED_DATASET = 'lastfm'
FIXED_X_EPS = 0.5
FIXED_EDGE_STRATEGY = '1-hop-x-to-1'

# X-Axis: Epsilon_Y to scan
X_TICKS_VALS = [1.5, 2.0, 2.5]
X_TICK_LABELS = ['1.5', '2.0', '2.5']

# Data Path
DATA_DIR = './final_result/data/ablation/'

# Bar Configuration
# hatche patterns: '/' (diagonal), '.' (dots), 'x' (cross), etc.
BAR_WIDTH = 0.25  # Width of individual bars
configs = [
    {
        'name': 'Random',
        'label_popularity': 'random',
        'color': '#d63552',  # Blue
        'hatch': '///',  # Diagonal lines
        'idx': 0
    },
    {
        'name': 'Unpopular',
        'label_popularity': 'unpopular',
        'color': '#008906',  # Purple
        'hatch': '---',  # Dots
        'idx': 1
    },
    {
        'name': 'Popular',
        'label_popularity': 'popular',
        'color': '#619bc6',  # Red
        'hatch': 'xx',  # Crosses
        'idx': 2
    }
]


def load_data():
    """Load the specific lastfm CSV file."""
    path = os.path.join(DATA_DIR, f'{FIXED_DATASET}_label_popularity.csv')
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Standardize strings
            str_cols = ['dataset', 'mechanism', 'model', 'target_selection',
                        'poison_strategy', 'edge_strategy', 'label_popularity']
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].str.lower()
            print(f"Loaded {FIXED_DATASET}: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()
    else:
        print(f"Warning: File not found {path}")
        return pd.DataFrame()


def plot_bar_group(ax, df_subset):
    if df_subset.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    # Base X positions for the groups (0, 1, 2)
    indices = np.arange(len(X_TICKS_VALS))

    for config in configs:
        # Filter specific line/bar data
        query = (df_subset['label_popularity'] == config['label_popularity'])
        df_config = df_subset[query]

        # Extract Y values aligned to X_TICKS_VALS
        y_vals = []
        y_errs = []

        for x_val in X_TICKS_VALS:
            # Find row matching this y_eps
            row = df_config[df_config['y_eps'] == x_val]
            if not row.empty:
                y_vals.append(row.iloc[0]['attack_acc'])
                err = row.iloc[0]['attack_ci'] if 'attack_ci' in row.columns else 0
                y_errs.append(err)
            else:
                y_vals.append(0)
                y_errs.append(0)

        # Calculate offset position for this bar
        # config['idx'] is 0, 1, 2. Center is 1.
        # 0 -> -width, 1 -> 0, 2 -> +width
        offset = (config['idx'] - 1) * BAR_WIDTH

        # Plot Bar
        # edgecolor='white' helps hatch visibility
        ax.bar(indices + offset, y_vals,
               width=BAR_WIDTH,
               yerr=y_errs,
               label=config['name'],
               color=config['color'],
               hatch=config['hatch'],
               edgecolor='white',
               linewidth=1,
               capsize=3,  # Error bar caps
               error_kw={'ecolor': 'black', 'elinewidth': 1.5})  # Error bar style


def main():
    df = load_data()

    # 3 Rows (Models) x 3 Columns (Mechanisms)
    # [修改点1] sharex=False, 确保每个子图独立显示X轴
    fig, axes = plt.subplots(len(MODELS), len(MECHANISMS),
                             figsize=(FIGSIZE_X, FIGSIZE_Y),
                             sharex=False, sharey=False)

    # Adjust layout
    # 稍微增加了 hspace (0.2 -> 0.3) 以避免X轴标签和下一行标题重叠
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=0.23, wspace=0.08)

    for row_idx, model in enumerate(MODELS):
        for col_idx, mech in enumerate(MECHANISMS):
            ax = axes[row_idx, col_idx]

            # --- Title Logic ---
            # Columns (Top Row): Show Mechanism Name
            if row_idx == 0:
                ax.set_title(f"{mech.upper()}", fontsize=TITLE_FONTSIZE, fontweight='medium')

            # Rows (Left Column): Show Model Name on Y-axis label or text
            if not df.empty:
                # Filter Data
                mask = (
                        (df['model'] == model) &
                        (df['mechanism'] == mech) &
                        (df['x_eps'] == FIXED_X_EPS) &
                        (df['edge_strategy'] == FIXED_EDGE_STRATEGY)
                )
                df_subset = df[mask]
                plot_bar_group(ax, df_subset)

            # --- Axis Settings ---
            # X-Axis Ticks
            ax.set_xticks(np.arange(len(X_TICKS_VALS)))
            ax.set_xticklabels(X_TICK_LABELS, fontweight='medium', fontsize=TICKLABEL_FONTSIZE)

            # Grid
            ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1.5, alpha=1.0)
            ax.xaxis.grid(False)

            # Y-Axis Labels
            ax.tick_params(axis='y', labelsize=TICKLABEL_FONTSIZE)

            if col_idx == 0:
                # Row Label: MODEL Name + "Accuracy"
                ax.set_ylabel(f"{model.upper()}\nAccuracy", fontsize=FONTSIZE, fontweight='medium')
            else:
                ax.set_ylabel('')

            # [修改点2] 移除判断条件，让所有子图都显示X轴标签
            ax.set_xlabel(r'$\epsilon_y$', fontsize=X_LABEL_FONTSIZE, fontweight='medium')

    # --- Legend ---
    # Create custom handles for legend to show hatch patterns correctly
    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02),
                   ncol=3, fontsize=LEGEND_FONTSIZE,
                   frameon=False,
                   shadow=False,
                   borderpad=1)
    else:
        print("Warning: No legend handles found.")

    output_dir = './final_result/figure'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ablation_label_popularity_lastfm.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()