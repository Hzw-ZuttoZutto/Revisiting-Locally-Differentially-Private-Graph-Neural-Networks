from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from feature_rewrite import load_repo_local_feature_rewrite_api


def test_repo_local_loader_binds_to_current_checkout():
    api = load_repo_local_feature_rewrite_api()
    helper_root = repo_root.resolve()
    module_path = Path(api.module.__file__).resolve()

    assert helper_root in module_path.parents
    assert helper_root / 'src' in module_path.parents
    assert hasattr(api, 'rewrite_features')
