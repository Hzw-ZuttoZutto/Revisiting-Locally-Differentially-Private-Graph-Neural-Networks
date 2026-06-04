#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

TARGET = Path("/data/hzw/Rethinking_DP_GNN_runtime/scripts/plot_cora_alpha_epsilon_m.py")
runpy.run_path(str(TARGET), run_name="__main__")
