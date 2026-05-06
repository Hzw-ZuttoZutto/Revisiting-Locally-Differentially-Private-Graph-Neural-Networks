import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SUBMODULE_FEATURE_REWRITE_ROOT = REPO_ROOT / 'submodule' / 'artificial-node-feature_generator'
SUBMODULE_FEATURE_REWRITE_HELPER = SUBMODULE_FEATURE_REWRITE_ROOT / 'feature_rewrite.py'

_submodule_feature_rewrite_api = None


def load_submodule_feature_rewrite_api():
    global _submodule_feature_rewrite_api

    if _submodule_feature_rewrite_api is not None:
        return _submodule_feature_rewrite_api

    if not SUBMODULE_FEATURE_REWRITE_HELPER.is_file():
        raise RuntimeError(
            f'artificial-node-feature_generator helper not found at {SUBMODULE_FEATURE_REWRITE_HELPER}. '
            'Expected repo-local submodule helper file.'
        )

    module_name = '_repo_local_artificial_node_feature_rewrite_helper'
    spec = importlib.util.spec_from_file_location(module_name, SUBMODULE_FEATURE_REWRITE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to build import spec for {SUBMODULE_FEATURE_REWRITE_HELPER}.')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'load_repo_local_feature_rewrite_api'):
        raise RuntimeError(
            'artificial-node-feature_generator helper is missing expected '
            'load_repo_local_feature_rewrite_api() entrypoint.'
        )

    _submodule_feature_rewrite_api = module.load_repo_local_feature_rewrite_api()
    return _submodule_feature_rewrite_api
