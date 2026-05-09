import importlib
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch_geometric.utils import subgraph
from mechanisms import (
    HighDimSquareWave,
    MultiBit,
    MultiDimPiecewise,
    PerturbationState,
    _multibit_kappa,
    _piecewise_stable_terms,
    supported_feature_mechanisms,
)
from utils import str2bool


REPO_ROOT = Path(__file__).resolve().parent
SUBMODULE_FEATURE_REWRITE_ROOT = REPO_ROOT / 'submodule' / 'artificial-node-feature_generator'
SUBMODULE_FEATURE_REWRITE_SRC = SUBMODULE_FEATURE_REWRITE_ROOT / 'src'
_repo_local_feature_rewrite_module = None


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


def _load_repo_local_feature_rewrite_module():
    global _repo_local_feature_rewrite_module

    if _repo_local_feature_rewrite_module is not None:
        return _repo_local_feature_rewrite_module

    if not SUBMODULE_FEATURE_REWRITE_ROOT.is_dir():
        raise RuntimeError(
            f'artificial-node-feature-generator repository root not found at {SUBMODULE_FEATURE_REWRITE_ROOT}. '
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

    _repo_local_feature_rewrite_module = module
    return _repo_local_feature_rewrite_module


class FeatureTransform:
    rewrite_common_arg_names = ()
    rewrite_supported_features = [
        'random_normal',
        'random_signed_onehot',
        'shared',
        'node_degree',
        'degree_bucket_range',
        'degree_bucket_distribution',
        'pagerank',
        'operator',
        'eigen',
        'eigen_norm',
        'deepwalk',
    ]
    supported_features = ['raw', 'sim', *rewrite_supported_features]
    supported_sim_mechanisms = {'mbm', 'pm', 'hds'}
    sim_reference_mechanisms = {
        'mbm': MultiBit,
        'pm': MultiDimPiecewise,
        'hds': HighDimSquareWave,
    }
    rewrite_feature_arg_names = {
        'random_normal': ('feature_dim', 'random_normal_mean', 'random_normal_std'),
        'random_signed_onehot': ('feature_dim',),
        'shared': ('feature_dim', 'shared_value'),
        'node_degree': ('feature_dim',),
        'degree_bucket_range': ('feature_dim', 'degree_bucket_num_buckets', 'degree_bucket_range_max'),
        'degree_bucket_distribution': ('feature_dim', 'degree_bucket_num_buckets'),
        'pagerank': ('feature_dim',),
        'operator': (),
        'eigen': ('feature_dim',),
        'eigen_norm': ('feature_dim',),
        'deepwalk': (
            'feature_dim',
            'deepwalk_walk_length',
            'deepwalk_number_walks',
            'deepwalk_window_size',
            'deepwalk_workers',
            'deepwalk_undirected',
        ),
    }
    all_rewrite_arg_names = tuple(
        dict.fromkeys(
            list(rewrite_common_arg_names)
            + [
                arg_name
                for arg_names in rewrite_feature_arg_names.values()
                for arg_name in arg_names
            ]
        )
    )

    def __init__(self, feature: dict(help='feature transformation method',
                                     choices=supported_features, option='-f') = 'raw',
                 sim_reference_eps: dict(help='reference epsilon used by sim for mechanism-aware m=best resolution and rectify',
                                         type=float) = None,
                 mechanism='mbm',
                 m='best',
                 x_eps=np.inf,
                 norm=False,
                 norm_scale='none',
                 feature_dim: dict(
                     help='feature dimension for artificial rewrite features (active only when --feature is neither raw nor sim)',
                     type=int,
                 ) = None,
                 scale: dict(
                     help='global multiplicative scaling applied immediately before the GNN (defaults to 1.0 when omitted)',
                     type=float,
                 ) = None,
                 feature_preprojection: dict(
                     help='enable a learnable Linear -> SELU -> Dropout preprojection before scale when --feature operator is selected',
                 ) = False,
                 preprojection_output_dim: dict(
                     help='output dimension for --feature-preprojection with --feature operator',
                     type=int,
                 ) = None,
                 random_normal_mean: dict(help='mean for --feature random_normal', type=float) = None,
                 random_normal_std: dict(help='standard deviation for --feature random_normal', type=float) = None,
                 shared_value: dict(help='fill value for --feature shared', type=float) = None,
                 degree_bucket_num_buckets: dict(
                     help='number of buckets for degree bucket rewrite features',
                     type=int,
                 ) = None,
                 degree_bucket_range_max: dict(
                     help='maximum degree range for --feature degree_bucket_range',
                     type=int,
                 ) = None,
                 deepwalk_walk_length: dict(help='walk length for --feature deepwalk', type=int) = None,
                 deepwalk_number_walks: dict(help='number of walks per node for --feature deepwalk', type=int) = None,
                 deepwalk_window_size: dict(help='window size for --feature deepwalk', type=int) = None,
                 deepwalk_workers: dict(help='worker count for --feature deepwalk', type=int) = None,
                 deepwalk_undirected: dict(
                     help='whether deepwalk should treat the graph as undirected',
                     type=str2bool,
                 ) = None,
                 x_steps=0):

        self.feature = feature
        self.sim_reference_eps = sim_reference_eps
        self.mechanism = str(mechanism).strip().lower()
        self.m = m
        self.x_eps = x_eps
        self.norm = bool(norm)
        self.norm_scale = norm_scale
        self.feature_dim = feature_dim
        self.scale = scale
        self.feature_preprojection = bool(feature_preprojection)
        self.preprojection_output_dim = preprojection_output_dim
        self.random_normal_mean = random_normal_mean
        self.random_normal_std = random_normal_std
        self.shared_value = shared_value
        self.degree_bucket_num_buckets = degree_bucket_num_buckets
        self.degree_bucket_range_max = degree_bucket_range_max
        self.deepwalk_walk_length = deepwalk_walk_length
        self.deepwalk_number_walks = deepwalk_number_walks
        self.deepwalk_window_size = deepwalk_window_size
        self.deepwalk_workers = deepwalk_workers
        self.deepwalk_undirected = deepwalk_undirected
        self.x_steps = x_steps
        self._rewrite_seed = None

    def set_rewrite_seed(self, seed):
        self._rewrite_seed = None if seed is None else int(seed)
        return self

    @staticmethod
    def _validate_sim_reference_eps(sim_reference_eps):
        if sim_reference_eps is None:
            raise ValueError('sim_reference_eps must be provided.')
        if not np.isfinite(sim_reference_eps) or sim_reference_eps <= 0:
            raise ValueError(f'sim_reference_eps must be finite and > 0, got {sim_reference_eps}.')
        return float(sim_reference_eps)

    @staticmethod
    def _sample_sim_mask(x, m):
        sampled_indices = torch.rand_like(x).topk(m, dim=1).indices
        sampled_mask = torch.zeros_like(x, dtype=torch.bool).scatter(1, sampled_indices, True)
        return sampled_mask

    @staticmethod
    def _sample_sim_rademacher(x):
        sampled_signs = torch.randint(2, x.size(), device=x.device)
        return (sampled_signs * 2 - 1).to(dtype=x.dtype)

    @staticmethod
    def _sample_sim_uniform(x, low, high):
        span = float(high) - float(low)
        return torch.rand_like(x) * span + float(low)

    def _resolve_sim_mechanism_name(self):
        mechanism_name = str(self.mechanism).strip().lower()
        if mechanism_name not in self.supported_sim_mechanisms:
            raise ValueError(
                f'feature=sim currently supports mechanism in {sorted(self.supported_sim_mechanisms)}, '
                f'got {mechanism_name}.'
            )
        return mechanism_name

    def _resolve_sim_reference_mechanism(self, num_features, apply_norm_config=False):
        if num_features < 1:
            raise ValueError('sim feature mode requires feature dimension >= 1.')

        reference_eps = self._validate_sim_reference_eps(self.sim_reference_eps)
        sim_mechanism = self._resolve_sim_mechanism_name()
        mechanism_cls = self.sim_reference_mechanisms[sim_mechanism]

        mechanism_kwargs = {
            'eps': reference_eps,
            'input_range': (0.0, 1.0),
            'm': self.m,
            'norm': False,
        }
        if apply_norm_config:
            mechanism_kwargs['norm'] = self.norm
            mechanism_kwargs['norm_scale'] = self.norm_scale
        mechanism = mechanism_cls(**mechanism_kwargs)
        resolved_m = mechanism._resolve_m(num_features)
        return sim_mechanism, mechanism, resolved_m

    def _build_sim_features(self, x):
        sim_mechanism, mechanism, resolved_m = self._resolve_sim_reference_mechanism(
            num_features=x.size(1),
            apply_norm_config=self.norm,
        )
        sampled_mask = self._sample_sim_mask(x, resolved_m)

        eps_per_dim = mechanism.eps / resolved_m
        if sim_mechanism == 'mbm':
            y = self._sample_sim_rademacher(x)
            kappa = _multibit_kappa(float(eps_per_dim))
            unit_support_scale = 1.0
        elif sim_mechanism == 'pm':
            y = self._sample_sim_uniform(x, low=-1.0, high=1.0)
            kappa, _, _, _, _ = _piecewise_stable_terms(float(eps_per_dim))
            unit_support_scale = 1.0
        elif sim_mechanism == 'hds':
            y = self._sample_sim_uniform(x, low=-2.0, high=2.0)
            one_minus_q = -math.expm1(-float(eps_per_dim))
            kappa = 1.0 - one_minus_q / float(eps_per_dim)
            unit_support_scale = 2.0
        else:
            raise ValueError(
                f'Unsupported sim mechanism {sim_mechanism}; expected one of '
                f'{sorted(self.supported_sim_mechanisms)}.'
            )

        z = sampled_mask.to(dtype=x.dtype) * y
        effective_gain = (resolved_m / x.size(1)) * kappa
        state = PerturbationState(
            z=z,
            d=int(x.size(1)),
            m=int(resolved_m),
            eps_per_dim=float(eps_per_dim),
            kappa=float(kappa),
            effective_gain=float(effective_gain),
            unit_support_scale=float(unit_support_scale),
        )
        mechanism._update_diagnostics(state)
        x_unit_hat = mechanism.rectify(state)
        sim_x = x_unit_hat
        if self.norm:
            sim_x = mechanism.normalize_output(sim_x,state)
        sim_x = sim_x.to(dtype=x.dtype)
        mechanism.output_range = mechanism.compute_output_range(state)

        return sim_x, mechanism.output_range

    def refresh_sim_features(self, data):
        if self.feature != 'sim':
            raise ValueError('refresh_sim_features is only available when --feature sim is active.')

        data.x, data.output_range = self._build_sim_features(data.x)
        return data

    def _build_rewrite_params(self):
        params = {}
        if self.feature_dim is not None:
            params['feature_dim'] = int(self.feature_dim)

        if self.feature == 'random_normal':
            if self.random_normal_mean is not None:
                params['mean'] = float(self.random_normal_mean)
            if self.random_normal_std is not None:
                params['std'] = float(self.random_normal_std)
        elif self.feature == 'shared':
            if self.shared_value is not None:
                params['value'] = float(self.shared_value)
        elif self.feature == 'degree_bucket_range':
            if self.degree_bucket_num_buckets is not None:
                params['num_buckets'] = int(self.degree_bucket_num_buckets)
            if self.degree_bucket_range_max is not None:
                params['range_max'] = int(self.degree_bucket_range_max)
        elif self.feature == 'degree_bucket_distribution':
            if self.degree_bucket_num_buckets is not None:
                params['num_buckets'] = int(self.degree_bucket_num_buckets)
        elif self.feature == 'deepwalk':
            if self.deepwalk_walk_length is not None:
                params['walk_length'] = int(self.deepwalk_walk_length)
            if self.deepwalk_number_walks is not None:
                params['number_walks'] = int(self.deepwalk_number_walks)
            if self.deepwalk_window_size is not None:
                params['window_size'] = int(self.deepwalk_window_size)
            if self.deepwalk_workers is not None:
                params['workers'] = int(self.deepwalk_workers)
            if self.deepwalk_undirected is not None:
                params['undirected'] = bool(self.deepwalk_undirected)
        elif self.feature == 'operator':
            params['x_steps'] = int(self.x_steps)

        return params

    def _rewrite_features(self, data):
        rewrite_module = _load_repo_local_feature_rewrite_module()
        return rewrite_module.rewrite_features(
            data,
            feature=self.feature,
            params=self._build_rewrite_params(),
            seed=self._rewrite_seed,
        )

    def __call__(self, data):
        if self.feature == 'sim':
            return self.refresh_sim_features(data)
        if self.feature in self.rewrite_supported_features:
            return self._rewrite_features(data)

        return data


class FeaturePerturbation:
    def __init__(self,
                 mechanism:     dict(help='feature perturbation mechanism', choices=list(supported_feature_mechanisms),
                                     option='-m') = 'mbm',
                 x_eps:         dict(help='privacy budget for feature perturbation', type=float,
                                     option='-ex') = np.inf,
                 m:             dict(help='sampled feature dimensions for MBM/PM/HDS (best, max, or positive integer)') = 'best',
                 norm:          dict(help='enable mechanism-internal post-perturbation feature normalization') = False,
                 norm_scale:    dict(
                     help='four-stage norm scaling: use "none" to project back into perturbation space, or a positive float for 1/norm_scale',
                     type=str,
                 ) = 'none',
                 inf_eps_unit_map: dict(
                     help='when x_eps=inf for four-stage mechanisms, deterministically map features to the mechanism unit domain [-1,1] for strict finite-vs-inf comparisons',
                 ) = False,
                 feature='raw',
                 data_range=None):

        self.mechanism = mechanism
        self.input_range = data_range
        self.x_eps = x_eps
        self.m = m
        self.norm = norm
        self.norm_scale = norm_scale
        self.inf_eps_unit_map = bool(inf_eps_unit_map)
        self.feature = feature

    def _resolve_input_range(self, data):
        input_range = self.input_range
        if input_range is None:
            input_range = (data.x.min().item(), data.x.max().item())

        alpha, beta = float(input_range[0]), float(input_range[1])
        if not np.isfinite(alpha) or not np.isfinite(beta) or beta <= alpha:
            raise ValueError(f'Invalid feature input range ({alpha}, {beta}); expected finite lower < upper.')
        return alpha, beta

    @staticmethod
    def _map_to_unit_interval(x, alpha, beta):
        return (x - alpha) * 2.0 / (beta - alpha) - 1.0

    def __call__(self, data):
        data.feature_mechanism = self.mechanism
        if self.feature != 'raw':
            if not hasattr(data, 'output_range'):
                data.output_range = None
            return data

        data.output_range = None
        if np.isinf(self.x_eps):
            if self.inf_eps_unit_map and self.mechanism in {'mbm', '1bm', 'pm', 'hds'}:
                alpha, beta = self._resolve_input_range(data)
                data.x = self._map_to_unit_interval(data.x, alpha=alpha, beta=beta)  # 做额外新型变化到堆成区间[-1,1]
            return data

        if not np.isinf(self.x_eps):
            input_range = self._resolve_input_range(data) # [0，1]
            mechanism_kwargs = {
                'eps': self.x_eps,
                'input_range': input_range,
                'norm': self.norm,
                'norm_scale': self.norm_scale,
            }
            if self.mechanism in {'mbm', 'pm', 'hds'}: # 如果是四阶段机制，需要解析下采样维度参数m
                mechanism_kwargs['m'] = self.m
            mechanism = supported_feature_mechanisms[self.mechanism](**mechanism_kwargs)
            # 为data 在这里添加一个属性 data.output_range = mechanism.output_range,mechanism当中我期望计算output_range 的方法由集成的类重写
            data.x = mechanism(data.x)
            data.output_range = mechanism.output_range

        return data


class NFR:
    def __init__(self, B, tao2):
        self.B = B
        self.tao2 = tao2

    def __call__(self, data):
        mu2 = self.B * self.tao2
        data.x = torch.sign(data.x) * torch.clamp(data.x.abs() - mu2, min=0.0)
        return data


class Normalize:
    def __init__(self, low, high):
        self.min = low
        self.max = high

    def __call__(self, data):
        alpha = data.x.min(dim=0)[0]
        beta = data.x.max(dim=0)[0]
        delta = beta - alpha
        scaled = torch.empty_like(data.x)
        non_constant = delta > 0
        if bool(non_constant.any().item()):
            scaled[:, non_constant] = (
                (data.x[:, non_constant] - alpha[non_constant])
                * (self.max - self.min)
                / delta[non_constant]
                + self.min
            )
        if bool((~non_constant).any().item()):
            scaled[:, ~non_constant] = self.min
        data.x = scaled
        return data


class FilterTopClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        y = torch.nn.functional.one_hot(data.y)
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:self.num_classes]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        data.num_nodes = data.y.size(0)

        if 'adj_t' in data:
            data.adj_t = data.adj_t[idx, idx]
        elif 'edge_index' in data:
            data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]

        return data
