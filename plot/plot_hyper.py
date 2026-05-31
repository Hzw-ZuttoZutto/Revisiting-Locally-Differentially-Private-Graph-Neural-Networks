import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Visual Style Settings (保持与 plot_overall 一致) ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

TITLE_FONTSIZE = 18
FONTSIZE = 14
X_LABEL_FONTSIZE = 16  # 大号横轴标签
LEGEND_FONTSIZE = 18
TICKLABEL_FONTSIZE = 12
LINEWIDTH = 2
MARKERSIZE = 10
FIGSIZE_X = 20
FIGSIZE_Y = 10
BOTTOM = 0.15
TOP = 0.92
LEFT = 0.06
RIGHT = 0.98

# Markers and Colors (完整列表)
# 0: Square(Blue), 1: TriangleUp(Purple), 2: TriangleDown(Red), 3: Circle(Green), 4: X(Yellow)
marker_list = ["s", "^", "o", "x", "v"]
color_list = ["#5372ab", "#936bb9", "#6aa56e", "#c9b97d", "#b75555"]

# --- Configuration ---
DATASETS = ['cora', 'pubmed', 'facebook', 'lastfm']
MECHANISMS = ['mbm', 'pm', 'sr']  # 3行
FIXED_MODEL = 'gat'
FIXED_X_EPS = 1.0  # 超参数实验固定 x_eps
FIXED_Y_EPS = 1.0  # 超参数实验固定 y_eps

# 数据路径
DATA_DIR = './final_result/data/hyper/'

# Attack Configurations (4条线)
# style_idx 用于确保颜色和标记与 Overall 图保持一致
# 0: No Attack (Blue)
# 2: Our Attack 0-hop (Red) - 跳过 1 (Baseline)
# 3: Our Attack 1-to-1 (Green)
# 4: Our Attack x-to-1 (Yellow)
configs = [
    {'name': 'No Attack', 'use_baseline': True, 'style_idx': 0},
    {'name': 'SIGMA-DI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '0-hop',
     'style_idx': 2},
    {'name': 'SIGMA-LNI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '1-hop-1-to-1',
     'style_idx': 3},
    {'name': 'SIGMA-GNI', 'use_baseline': False, 'target_selection': 'high-influence',
     'poison_strategy': 'maximal-difference-based', 'label_popularity': 'popular', 'edge_strategy': '1-hop-x-to-1',
     'style_idx': 4}
]


def load_all_data():
    data_map = {}
    for d_name in DATASETS:
        path = os.path.join(DATA_DIR, f'{d_name}.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # 标准化字符串
                df['dataset'] = df['dataset'].str.lower()
                df['mechanism'] = df['mechanism'].str.lower()
                df['model'] = df['model'].str.lower()
                data_map[d_name] = df
                print(f"Loaded {d_name}: {df.shape}")
            except Exception as e:
                print(f"Error loading {d_name}: {e}")
                data_map[d_name] = pd.DataFrame()
        else:
            print(f"Warning: File not found {path}")
            data_map[d_name] = pd.DataFrame()
    return data_map


def plot_lines(ax, df_subset, beta_mapping):
    """
    修改后的 plot_lines 增加了一个参数 beta_mapping。
    beta_mapping 是一个字典，将真实的 beta 值映射为 0, 1, 2... 的索引。
    """
    if df_subset.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    for config in configs:
        style_idx = config['style_idx']  # 获取对应的样式索引

        if config['use_baseline']:
            # No Attack: Group by influence_ratio
            df_grouped = df_subset.groupby('influence_ratio').first().reset_index()
            df_sorted = df_grouped.sort_values('influence_ratio')
            x_vals_raw = df_sorted['influence_ratio']
            y_vals = df_sorted['baseline_acc']
            err_vals = df_sorted.get('baseline_ci', [0] * len(y_vals))
        else:
            # Attacks
            query_parts = []
            for k, v in config.items():
                if k not in ['name', 'use_baseline', 'style_idx']:
                    if isinstance(v, str):
                        query_parts.append(f"{k} == '{v}'")
                    else:
                        query_parts.append(f"{k} == {v}")
            query = ' & '.join(query_parts)
            df_config = df_subset.query(query)

            if df_config.empty:
                continue

            df_sorted = df_config.sort_values('influence_ratio')
            x_vals_raw = df_sorted['influence_ratio']
            y_vals = df_sorted['attack_acc']
            err_vals = df_sorted.get('attack_ci', [0] * len(y_vals))

        # [逻辑修改]：将原始的 x 值转换为索引，实现均匀分布
        # 过滤掉不在 mapping 中的异常值（如果有的话）
        valid_indices = [i for i, x in enumerate(x_vals_raw) if x in beta_mapping]
        if not valid_indices:
            continue

        x_plot = [beta_mapping[x_vals_raw.iloc[i]] for i in valid_indices]
        y_plot = y_vals.iloc[valid_indices]
        err_plot = list(np.array(err_vals)[valid_indices])

        ax.errorbar(x_plot, y_plot, yerr=err_plot,
                    label=config['name'],
                    color=color_list[style_idx],  # 使用指定颜色
                    marker=marker_list[style_idx],  # 使用指定标记
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH, capsize=0,
                    markerfacecolor='none', markeredgewidth=2)

        # [逻辑修改]：原先的 ax.set_xticks 移到 main 函数中统一处理，这里不再单独设置


def main():
    data_map = load_all_data()

    # [逻辑修改]：预先扫描所有数据，获取所有唯一的 influence_ratio 并排序，用于建立全局统一的均匀刻度
    all_betas = set()
    for df in data_map.values():
        if not df.empty and 'influence_ratio' in df.columns:
            # 简单过滤下相关数据，或者直接取全量 unique 也可以，这里为了严谨稍微过滤
            mask = (df['model'] == FIXED_MODEL) & \
                   (df['x_eps'] == FIXED_X_EPS) & \
                   (df['y_eps'] == FIXED_Y_EPS)
            subset = df[mask]
            if not subset.empty:
                all_betas.update(subset['influence_ratio'].unique())

    # 排序后的 beta 列表
    sorted_betas = sorted(list(all_betas))
    if not sorted_betas:
        # 如果没读到数据，给一个默认值防止报错
        sorted_betas = [0.01, 0.05, 0.07, 0.1, 0.15]

    # 建立映射字典：值 -> 索引
    beta_mapping = {val: i for i, val in enumerate(sorted_betas)}

    # 3行 x 4列
    fig, axes = plt.subplots(len(MECHANISMS), len(DATASETS),
                             figsize=(FIGSIZE_X, FIGSIZE_Y),
                             sharey=False, sharex=False)

    # 布局调整（紧凑）
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, hspace=0.25, wspace=0.1)

    for row_idx, mech in enumerate(MECHANISMS):
        for col_idx, d_name in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]

            # --- 标题设置 (仅第一行) ---
            if row_idx == 0:
                letter = chr(97 + col_idx)  # a, b, c, d
                formatted_name = d_name.title() if d_name != 'lastfm' else 'LastFM'
                if d_name == 'lastfm':
                    formatted_name = 'LastFM'
                else:
                    formatted_name = d_name.title()
                ax.set_title(f"({letter}) {formatted_name}", fontsize=TITLE_FONTSIZE, fontweight='medium')

            df = data_map.get(d_name, pd.DataFrame())

            if not df.empty:
                # 筛选数据：固定模型、机制、X_EPS, Y_EPS
                mask = (
                        (df['model'] == FIXED_MODEL) &
                        (df['mechanism'] == mech) &
                        (df['x_eps'] == FIXED_X_EPS) &
                        (df['y_eps'] == FIXED_Y_EPS)
                )
                df_subset = df[mask]
                plot_lines(ax, df_subset, beta_mapping)

            # --- 坐标轴美化 ---

            # [逻辑修改]：设置均匀刻度
            # 刻度位置设为 0, 1, 2...
            ax.set_xticks(range(len(sorted_betas)))
            # 刻度标签设为原始 beta 值
            ax.set_xticklabels(sorted_betas)

            # 横轴标签：beta
            ax.set_xlabel(r'$\eta$', fontsize=X_LABEL_FONTSIZE, fontweight='medium')

            # 网格线
            ax.grid(True, color='white', linestyle='-', linewidth=1, alpha=1.0)

            # 刻度样式 (保持原代码样式)
            ax.tick_params(axis='both', which='major', labelsize=TICKLABEL_FONTSIZE)

            # 纵轴标签 (仅第一列)
            if col_idx == 0:
                ax.set_ylabel(f"{mech.upper()}\nAccuracy", fontsize=FONTSIZE, fontweight='medium')
            else:
                ax.set_ylabel('')

    # --- 图例 ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02),
                   ncol=4,  # 4个图例项
                   fontsize=LEGEND_FONTSIZE,
                   frameon=False,
                   shadow=False,
                   borderpad=1)
    else:
        print("Warning: No legend handles found.")

    output_dir = './final_result/figure'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'hyper_gat.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()