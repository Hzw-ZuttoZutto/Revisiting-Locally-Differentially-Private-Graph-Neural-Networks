from __future__ import annotations

import math

import matplotlib.pyplot as plt

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
        assert axes.shape == (3, 2)
        assert tuple(fig.get_size_inches()) == (12.0, 11.5)
        assert math.isclose(fig.subplotpars.bottom, 0.19)
        assert math.isclose(fig.subplotpars.top, 0.92)
        assert math.isclose(fig.subplotpars.left, 0.10)
        assert math.isclose(fig.subplotpars.right, 0.99)
        assert math.isclose(fig.subplotpars.hspace, 0.44)
        assert math.isclose(fig.subplotpars.wspace, 0.16)
        assert axes[0, 0].get_title() == "(a) Actor"
        assert axes[0, 1].get_title() == "(b) Flickr"
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
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        assert legend._ncols == 3
        assert [text.get_text() for text in legend.get_texts()] == list(
            plotter.LINE_ORDER
        )
        for axis in axes.flat:
            assert len(axis.lines) == len(plotter.LINE_ORDER)
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
