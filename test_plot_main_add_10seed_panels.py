from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts import plot_main_add_10seed_panels as plotter


def synthetic_plot_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backbone_index, backbone in enumerate(plotter.BACKBONES):
        for dataset_index, dataset in enumerate(plotter.DATASETS):
            for line_index, label in enumerate(plotter.LINE_ORDER):
                for x_index, x_eps in enumerate(plotter.X_EPS_VALUES):
                    mean = 60.0 + backbone_index + dataset_index + line_index + x_index * 0.1
                    rows.append(
                        {
                            "source": (
                                "featfree_homo_rerun_hoa"
                                if label == plotter.FEATFREE_LABEL
                                else "style_test"
                            ),
                            "backbone": backbone,
                            "dataset": dataset,
                            "x_eps": x_eps,
                            "x_index": x_index,
                            "line_label": label,
                            "test_acc_mean": mean,
                            "test_acc_ci_low": mean - 0.2,
                            "test_acc_ci_high": mean + 0.2,
                            "n": 10,
                        }
                    )
    return rows


def test_frozen_style_contract() -> None:
    assert plotter.FROZEN_STYLE_VERSION == "paper_figure1_reference_v1"
    assert plotter.DEFAULT_OUTPUT_DIR == plotter.REPO_ROOT / "rebuttal_figure"
    assert plotter.OUTPUT_STEM == "figure1"
    assert plotter.PLOT_DATA_FILENAME == "figure1_plot_data.csv"
    assert (plotter.FIGSIZE_X, plotter.FIGSIZE_Y) == (20, 11.5)
    assert (
        plotter.TITLE_FONTSIZE,
        plotter.FONTSIZE,
        plotter.X_LABEL_FONTSIZE,
        plotter.LEGEND_FONTSIZE,
        plotter.TICKLABEL_FONTSIZE,
    ) == (18, 18, 18, 24, 15)
    assert (plotter.BOTTOM, plotter.TOP, plotter.LEFT, plotter.RIGHT) == (
        0.14,
        0.92,
        0.065,
        0.99,
    )
    assert (plotter.HSPACE, plotter.WSPACE) == (0.34, 0.16)
    assert (plotter.TITLE_PAD, plotter.Y_LABEL_PAD) == (18, 14)
    assert plotter.LEGEND_BBOX == (0.5, 0.005)
    assert (
        plotter.LEGEND_BORDERPAD,
        plotter.LEGEND_HANDLELENGTH,
        plotter.LEGEND_COLUMNSPACING,
    ) == (0.2, 1.3, 0.75)
    assert (plotter.LINEWIDTH, plotter.MARKERSIZE, plotter.MARKEREDGEWIDTH) == (2, 8, 2)
    assert plotter.SAVE_DPI == 300
    assert plotter.SAVE_PAD_INCHES == 0.02
    assert plotter.LINE_ORDER == (
        r"$\mathsf{LPGNN}$",
        r"$\mathsf{PrivGE}$",
        r"$\mathsf{UPGNet\text{-}MBM}$",
        r"$\mathsf{UPGNet\text{-}PM}$",
        r"$\mathsf{FeatFree}$",
        r"$\mathsf{Non\text{-}private}$",
    )
    expected_styles = (
        ("#5372ab", "s", "-"),
        ("#936bb9", "^", "-"),
        ("#6aa56e", "o", "-"),
        ("#c9b97d", "x", "-"),
        ("#b75555", "D", "--"),
        ("#2f7fb8", "P", "--"),
    )
    assert tuple(
        (
            plotter.STYLE_CONFIGS[label]["color"],
            plotter.STYLE_CONFIGS[label]["marker"],
            plotter.STYLE_CONFIGS[label]["linestyle"],
        )
        for label in plotter.LINE_ORDER
    ) == expected_styles


def test_frozen_figure_artists() -> None:
    fig, axes = plotter.build_figure(synthetic_plot_rows())
    try:
        assert axes.shape == (3, 4)
        np.testing.assert_allclose(fig.get_size_inches(), [20, 11.5])
        assert math.isclose(fig.subplotpars.bottom, 0.14)
        assert math.isclose(fig.subplotpars.top, 0.92)
        assert math.isclose(fig.subplotpars.left, 0.065)
        assert math.isclose(fig.subplotpars.right, 0.99)
        assert math.isclose(fig.subplotpars.hspace, 0.34)
        assert math.isclose(fig.subplotpars.wspace, 0.16)

        for column, dataset in enumerate(plotter.DATASETS):
            assert axes[0, column].get_title() == (
                f"({chr(97 + column)}) {plotter.DATASET_LABELS[dataset]}"
            )
            assert axes[0, column].title.get_fontsize() == 18
            assert axes[0, column].title.get_fontweight() == "medium"

        for row, backbone in enumerate(plotter.BACKBONES):
            assert axes[row, 0].get_ylabel() == (
                f"{plotter.BACKBONE_LABELS[backbone]}\nAccuracy"
            )
            assert axes[row, 0].yaxis.label.get_fontsize() == 18
            assert axes[row, 0].yaxis.labelpad == 14
            for column in range(1, len(plotter.DATASETS)):
                assert axes[row, column].get_ylabel() == ""

        axis = axes[0, 0]
        assert axis.get_xlabel() == r"$\epsilon$"
        assert axis.xaxis.label.get_fontsize() == 18
        assert [tick.get_text() for tick in axis.get_xticklabels()] == list(
            plotter.X_EPS_VALUES
        )
        assert all(tick.get_rotation() == 30 for tick in axis.get_xticklabels())
        assert all(tick.get_ha() == "right" for tick in axis.get_xticklabels())
        assert all(tick.get_fontsize() == 15 for tick in axis.get_xticklabels())

        assert len(axis.lines) == len(plotter.LINE_ORDER)
        assert [container.get_label() for container in axis.containers] == list(
            plotter.LINE_ORDER
        )
        for line, label in zip(axis.lines, plotter.LINE_ORDER, strict=True):
            style = plotter.STYLE_CONFIGS[label]
            assert line.get_color() == style["color"]
            assert line.get_marker() == style["marker"]
            assert line.get_linestyle() == style["linestyle"]
            assert line.get_markersize() == 8
            assert line.get_linewidth() == 2
            assert line.get_markerfacecolor() == "none"
            assert line.get_markeredgewidth() == 2

        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == list(
            plotter.LINE_ORDER
        )
        assert all(text.get_fontsize() == 24 for text in legend.get_texts())
        assert legend.get_frame_on() is False
        assert legend._ncols == 6
        assert legend._loc == 8
        assert legend.get_bbox_to_anchor()._bbox.bounds == (0.5, 0.005, 0.0, 0.0)
        assert legend.borderpad == 0.2
        assert legend.handlelength == 1.3
        assert legend.columnspacing == 0.75
    finally:
        plt.close(fig)


def test_featfree_homo_rerun_inputs_are_complete() -> None:
    plotter.validate_render_environment()
    choices = plotter.load_featfree_choices(plotter.DEFAULT_FEATFREE_ROOT)
    assert len(choices) == 12
    assert set(choices) == {
        (backbone, dataset)
        for backbone in plotter.BACKBONES
        for dataset in plotter.DATASETS
    }
    assert sum(len(choice.test_accs) for choice in choices.values()) == 120
    assert all(choice.source_job_dir.is_dir() for choice in choices.values())
    assert all(choice.smoother == "hoa" for choice in choices.values())


def test_real_plot_rows_have_frozen_shape() -> None:
    main_rows = plotter.load_main_curves(
        plotter.DEFAULT_MAIN_LONG_CSV,
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    featfree_rows = plotter.featfree_plot_rows(
        plotter.load_featfree_choices(plotter.DEFAULT_FEATFREE_ROOT),
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    clean_rows = plotter.clean_reference_plot_rows(
        plotter.DEFAULT_CLEAN_REFERENCE_MANIFEST,
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    assert (len(main_rows), len(featfree_rows), len(clean_rows)) == (480, 120, 120)
    rows = main_rows + featfree_rows + clean_rows
    plotter.validate_plot_rows(rows)
    assert len(rows) == 720
    assert {
        row["source"] for row in featfree_rows
    } == {"featfree_homo_rerun_hoa"}
