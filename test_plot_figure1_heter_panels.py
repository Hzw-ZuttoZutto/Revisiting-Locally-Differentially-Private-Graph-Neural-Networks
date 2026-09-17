from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from scripts import plot_figure1_heter_panels as plotter


def synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backbone_index, backbone in enumerate(plotter.BACKBONES):
        for dataset_index, dataset in enumerate(plotter.DATASETS):
            for line_index, label in enumerate(plotter.LINE_ORDER):
                for x_index, x_eps in enumerate(plotter.X_EPS_VALUES):
                    mean = 20.0 + backbone_index + dataset_index + line_index + x_index
                    rows.append(
                        {
                            "source": "synthetic",
                            "backbone": backbone,
                            "dataset": dataset,
                            "x_eps": x_eps,
                            "x_index": x_index,
                            "line_label": label,
                            "test_acc_mean": mean,
                            "test_acc_ci_low": mean - 0.1,
                            "test_acc_ci_high": mean + 0.1,
                            "n": 10,
                        }
                    )
    return rows


def test_heter_figure_style_contract() -> None:
    fig, axes = plotter.build_figure(synthetic_rows())
    try:
        assert plotter.PANEL_Y_TICKS == {
            ("gcn", "actor"): (25.0, 27.5, 30.0),
            ("gcn", "attributedgraph-flickr"): (25.0, 37.5, 50.0, 62.5),
            ("gat", "attributedgraph-flickr"): (20.0, 30.0, 40.0, 50.0),
        }
        assert axes.shape == (3, 2)
        assert tuple(fig.get_size_inches()) == (12.0, 11.5)
        assert math.isclose(fig.subplotpars.bottom, 0.19)
        assert math.isclose(fig.subplotpars.top, 0.92)
        assert math.isclose(fig.subplotpars.left, 0.10)
        assert math.isclose(fig.subplotpars.right, 0.99)
        assert math.isclose(fig.subplotpars.hspace, 0.44)
        assert math.isclose(fig.subplotpars.wspace, 0.16)
        assert axes[0, 0].get_title() == "(a) Actor (Majority Class: 25.86%)"
        assert axes[0, 1].get_title() == "(b) Flickr (Majority Class: 11.72%)"
        assert {
            axes[0, column].title.get_fontsize() for column in range(2)
        } == {plotter.PANEL_HEADING_FONTSIZE}
        assert [axes[index, 0].get_ylabel() for index in range(3)] == [
            "GCN",
            "GraphSAGE",
            "GAT",
        ]
        for axis in axes.flat:
            assert axis.xaxis.label.get_fontsize() == plotter.AXIS_LABEL_FONTSIZE
            assert axis.xaxis.labelpad == plotter.X_LABEL_PAD
            assert {
                label.get_fontsize() for label in axis.get_xticklabels()
            } == {plotter.TICKLABEL_FONTSIZE}
            assert {
                label.get_fontsize() for label in axis.get_yticklabels()
            } == {plotter.TICKLABEL_FONTSIZE}
        for axis in axes[:, 0]:
            assert axis.yaxis.label.get_fontsize() == plotter.PANEL_HEADING_FONTSIZE
            assert axis.yaxis.labelpad == plotter.BACKBONE_LABEL_PAD
            assert axis.yaxis.label.get_position() == (plotter.BACKBONE_LABEL_X, 0.5)
        for (backbone, dataset), expected_ticks in plotter.PANEL_Y_TICKS.items():
            row = plotter.BACKBONES.index(backbone)
            column = plotter.DATASETS.index(dataset)
            assert tuple(axes[row, column].get_yticks()) == expected_ticks
        assert tuple(axes[1, 0].get_yticks()) == (20.0, 25.0, 30.0, 35.0, 40.0)
        assert tuple(axes[1, 1].get_yticks()) == (10.0, 20.0, 30.0, 40.0)
        assert tuple(axes[2, 0].get_yticks()) == (20.0, 25.0, 30.0, 35.0, 40.0)
        assert len(fig.legends) == 2
        for legend, expected_row in zip(
            fig.legends,
            (plotter.LEGEND_TOP_ROW, plotter.LEGEND_BOTTOM_ROW),
        ):
            assert legend._ncols == len(expected_row)
            assert [text.get_text() for text in legend.get_texts()] == [
                plotter.LEGEND_DISPLAY_LABELS.get(label, label)
                for label in expected_row
            ]
            assert legend.get_bbox_to_anchor()._bbox.x0 == 0.5
        for column_index, dataset in enumerate(plotter.DATASETS):
            expected_baseline = plotter.MAJORITY_BASELINE_PCT[dataset]
            for axis in axes[:, column_index]:
                assert len(axis.lines) == len(plotter.LINE_ORDER) + 1
                baseline_line = axis.lines[-1]
                assert np.allclose(
                    np.asarray(baseline_line.get_ydata(), dtype=float),
                    expected_baseline,
                )
                assert baseline_line.get_color() == plotter.MAJORITY_BASELINE_COLOR
                assert (
                    baseline_line.get_linestyle()
                    == plotter.MAJORITY_BASELINE_LINESTYLE
                )
                assert baseline_line.get_linewidth() == plotter.MAJORITY_BASELINE_LINEWIDTH
                assert baseline_line.get_marker() == "None"
                assert baseline_line.get_label() == plotter.MAJORITY_LABEL
                assert axis.get_ylim()[0] <= expected_baseline <= axis.get_ylim()[1]
                assert list(axis.texts) == []
        for axis in axes.flat:
            assert [container.get_label() for container in axis.containers] == list(
                plotter.LINE_ORDER
            )
    finally:
        plt.close(fig)


def test_real_heter_plot_rows_are_complete() -> None:
    main_rows = plotter.load_main_rows(
        plotter.DEFAULT_MAIN_ROOT,
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    featfree_rows = plotter.load_featfree_rows(
        plotter.DEFAULT_FEATFREE_ROOT,
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    clean_rows = plotter.load_clean_rows(
        plotter.DEFAULT_CLEAN_ROOT,
        bootstrap_samples=plotter.BOOTSTRAP_SAMPLES,
        bootstrap_seed=plotter.BOOTSTRAP_SEED,
    )
    assert (len(main_rows), len(featfree_rows), len(clean_rows)) == (240, 60, 60)
    rows = main_rows + featfree_rows + clean_rows
    plotter.validate_plot_rows(rows)
    assert len(rows) == 360
    assert {row["source"] for row in featfree_rows} == {
        "featfree_heter_hoa_random_projected"
    }
    assert {int(row["feature_dim"]) for row in featfree_rows if row["dataset"] == "actor"} == {
        932
    }
    assert {
        int(row["feature_dim"])
        for row in featfree_rows
        if row["dataset"] == "attributedgraph-flickr"
    } == {12047}
