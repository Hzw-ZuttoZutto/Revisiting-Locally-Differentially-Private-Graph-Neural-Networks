import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parent
SUBMODULE_FEATURE_REWRITE_ROOT = REPO_ROOT
SUBMODULE_FEATURE_REWRITE_SRC = SUBMODULE_FEATURE_REWRITE_ROOT / 'src'


@dataclass(frozen=True)
class RepoLocalFeatureRewriteAPI:
    module: ModuleType
    rewrite_features: Callable


_repo_local_feature_rewrite_api = None


def _module_origin_path(module_name, module):
    module_path = getattr(module, '__file__', None)
    if module_path is None:
        raise RuntimeError(f'Imported {module_name} has no __file__; unable to verify import origin.')
    return Path(module_path).resolve()


def _assert_module_under_submodule_root(module_name, module):
    module_path = _module_origin_path(module_name, module)
    submodule_root = SUBMODULE_FEATURE_REWRITE_ROOT.resolve()
    if module_path != submodule_root and submodule_root not in module_path.parents:
        raise RuntimeError(
            f'Imported {module_name} from unexpected location: {module_path}. '
            f'Expected path under {submodule_root}.'
        )


def _purge_modules(prefix):
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(f'{prefix}.'):
            del sys.modules[module_name]


def load_repo_local_feature_rewrite_api():
    global _repo_local_feature_rewrite_api

    if _repo_local_feature_rewrite_api is not None:
        return _repo_local_feature_rewrite_api

    if not SUBMODULE_FEATURE_REWRITE_ROOT.is_dir():
        raise RuntimeError(
            f'artificial-node-feature_generator repository root not found at {SUBMODULE_FEATURE_REWRITE_ROOT}. '
            'Please initialize/update the repository checkout first.'
        )
    if not SUBMODULE_FEATURE_REWRITE_SRC.is_dir():
        raise RuntimeError(
            f'artificial-node-feature-generator source directory not found at {SUBMODULE_FEATURE_REWRITE_SRC}. '
            'Expected a repo-local source checkout.'
        )

    submodule_src = str(SUBMODULE_FEATURE_REWRITE_SRC)
    if submodule_src in sys.path:
        sys.path.remove(submodule_src)
    sys.path.insert(0, submodule_src)

    _purge_modules('artificial_node_feature_generator')

    try:
        module = importlib.import_module('artificial_node_feature_generator')
    except Exception as exc:
        raise ImportError(
            'Failed to import artificial_node_feature_generator from repo-local source path. '
            f'Expected path: {SUBMODULE_FEATURE_REWRITE_SRC}'
        ) from exc

    _assert_module_under_submodule_root('artificial_node_feature_generator', module)
    if not hasattr(module, 'rewrite_features'):
        raise RuntimeError('artificial_node_feature_generator is missing expected attribute: rewrite_features')

    _repo_local_feature_rewrite_api = RepoLocalFeatureRewriteAPI(
        module=module,
        rewrite_features=module.rewrite_features,
    )
    return _repo_local_feature_rewrite_api
