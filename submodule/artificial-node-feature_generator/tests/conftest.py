from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
src_root_str = str(src_root)
if src_root_str in sys.path:
    sys.path.remove(src_root_str)
sys.path.insert(0, src_root_str)

for module_name in list(sys.modules):
    if module_name == "artificial_node_feature_generator" or module_name.startswith("artificial_node_feature_generator."):
        del sys.modules[module_name]
