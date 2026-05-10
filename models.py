import math

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Dropout, Linear, SELU
from torch_geometric.nn import MessagePassing, SAGEConv, GCNConv, GATConv
from torch_sparse import SparseTensor, matmul

try:
    from torch_geometric.utils import accuracy as accuracy_1d
except ImportError:
    def accuracy_1d(pred, target):
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (pred == target).float().mean()


class KProp(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, transform=lambda x: x):
        super().__init__(aggr=aggregator)
        self.transform = transform
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            self._cached_x = self.neighborhood_aggregation(x, adj_t)

        return self._cached_x

    def neighborhood_aggregation(self, x, adj_t):
        if self.K <= 0:
            return x

        if self.normalize:
            adj_t = gcn_norm(adj_t, add_self_loops=False)
        if self.add_self_loops:
            adj_t = adj_t.set_diag()

        for k in range(self.K): # K次消息传播
            x = self.propagate(adj_t, x=x)

        x = self.transform(x)
        return x

    def message_and_aggregate(self, adj_t, x):  #消息传播的方式
        return matmul(adj_t, x, reduce=self.aggr)


class HOA(MessagePassing):
    def __init__(self, steps, aggregator, add_self_loops, normalize, cached, transform=lambda x: x):
        super().__init__(aggr=aggregator)
        self.transform = transform
        self.K = steps
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self._cached_x = None

    def forward(self, x, adj_t):
        if self._cached_x is None or not self.cached:
            self._cached_x = self.neighborhood_aggregation(x, adj_t)

        return self._cached_x

    def neighborhood_aggregation(self, x, adj_t):
        if self.K <= 0:
            return x

        if self.normalize:
            adj_t = gcn_norm(adj_t, add_self_loops=False)
        if self.add_self_loops:
            adj_t = adj_t.set_diag()

        x_i = x
        h = torch.zeros_like(x)

        for _ in range(self.K):
            x_i = self.propagate(adj_t, x=x_i)
            h = h + x_i

        out = h / self.K
        out = self.transform(out)
        return out

    def message_and_aggregate(self, adj_t, x):  # noqa
        return matmul(adj_t, x, reduce=self.aggr)


class GNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = None
        self.conv2 = None
        self.dropout = Dropout(p=dropout)
        self.activation = SELU(inplace=True)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x


class GCN(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)


class GAT(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        heads = 4
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(heads * hidden_dim, output_dim, heads=1, concat=False)


class GraphSAGE(GNN):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout):
        super().__init__(dropout)
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=False, root_weight=True)
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim, normalize=False, root_weight=True)


class NodeClassifier(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 feature='raw',
                 model:                 dict(help='backbone GNN model', choices=['gcn', 'sage', 'gat']) = 'sage',
                 hidden_dim:            dict(help='dimension of the hidden layers') = 16,
                 dropout:               dict(help='dropout rate (between zero and one)') = 0.0,
                 scale=1.0,
                 feature_preprojection=False,
                 preprojection_output_dim=None,
                 x_steps:               dict(help='feature smoother step parameter', option='-kx') = 0,
                 smoother:              dict(help='feature smoother before GNN', choices=['kprop', 'hoa']) = 'kprop',
                 ):
        super().__init__()
        self.feature = str(feature).strip().lower()
        self.backbone = str(model).strip().lower()
        self.scale = self._resolve_feature_scale(scale)
        self.feature_preprojection = bool(feature_preprojection)
        self.preprojection_output_dim = preprojection_output_dim
        self.operator_x_steps = int(x_steps)

        if self.feature_preprojection and self.feature != 'operator':
            raise ValueError('feature_preprojection is only supported when feature="operator".')
        if not self.feature_preprojection and preprojection_output_dim is not None:
            raise ValueError('preprojection_output_dim requires feature_preprojection to be enabled.')
        if self.feature_preprojection:
            if preprojection_output_dim is None or int(preprojection_output_dim) <= 0:
                raise ValueError('preprojection_output_dim must be a positive integer when feature_preprojection is enabled.')
            self.feature_preprojection_layer = Linear(input_dim, int(preprojection_output_dim))
            self.feature_preprojection_activation = SELU(inplace=True)
            self.feature_preprojection_dropout = Dropout(p=dropout)
            gnn_input_dim = int(preprojection_output_dim)
        else:
            self.feature_preprojection_layer = None
            self.feature_preprojection_activation = None
            self.feature_preprojection_dropout = None
            gnn_input_dim = input_dim

        smoother_to_cls = {
            'kprop': KProp,
            'hoa': HOA,
        }
        if smoother not in smoother_to_cls:
            supported = sorted(smoother_to_cls)
            raise ValueError(f"Unsupported smoother {smoother!r}; expected one of {supported}.")

        self.smoother = smoother_to_cls[smoother](
            steps=x_steps,
            aggregator='add',
            add_self_loops=False, # LPGNN论文说去掉自环对于性能会更好
            normalize=True,
            cached=True,
        )

        self.gnn = {'gcn': GCN, 'sage': GraphSAGE, 'gat': GAT}[model](
            input_dim=gnn_input_dim,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self._cached_smoother_adj_id = None

    def clear_cached_state(self):
        self.smoother._cached_x = None
        self._cached_smoother_adj_id = None

    @staticmethod
    def _resolve_feature_scale(scale):
        resolved = float(scale)
        if not math.isfinite(resolved) or resolved <= 0:
            raise ValueError('scale must be > 0.')
        return resolved

    def _apply_scale(self, x):
        if self.scale == 1.0:
            return x
        return x * self.scale

    @staticmethod
    def _sparse_tensor_device(adj_t):
        row, _, value = adj_t.coo()
        if value is not None:
            return value.device
        return row.device

    @classmethod
    def _move_sparse_tensor(cls, adj_t, device):
        if cls._sparse_tensor_device(adj_t) == device:
            return adj_t
        row, col, value = adj_t.coo()
        if value is None:
            value = torch.ones(row.numel(), dtype=torch.float32, device=row.device)
        return SparseTensor(
            row=row.to(device),
            col=col.to(device),
            value=value.to(device=device, dtype=torch.float32),
            sparse_sizes=adj_t.sparse_sizes(),
        ).coalesce()

    @staticmethod
    def _graphsage_mean_aggregate(adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce='mean')

    @staticmethod
    def _apply_operator_to_matrix(operator_adj_t, matrix, *, x_steps):
        if x_steps <= 0:
            return matrix

        current = matrix
        accumulated = None
        for _ in range(int(x_steps)):
            current = matmul(operator_adj_t, current, reduce='add')
            accumulated = current if accumulated is None else accumulated + current
        return accumulated / float(x_steps)

    def _has_lazy_operator_input(self, data):
        return (
            self.feature == 'operator'
            and getattr(data, 'operator_feature_mode', None) == 'lazy_sparse'
            and hasattr(data, 'operator_normalized_adj_t')
        )

    def _operator_steps(self, data):
        return int(getattr(data, 'operator_x_steps', self.operator_x_steps))

    def _operator_adj_t(self, data, *, device):
        operator_adj_t = getattr(data, 'operator_normalized_adj_t', None)
        if operator_adj_t is None:
            raise ValueError('lazy operator input requires data.operator_normalized_adj_t')
        return self._move_sparse_tensor(operator_adj_t, device)

    def _apply_operator_projection(self, x, data=None):
        if self.feature_preprojection_layer is None:
            return x
        if data is not None and self._has_lazy_operator_input(data):
            operator_adj_t = self._operator_adj_t(
                data,
                device=self.feature_preprojection_layer.weight.device,
            )
            x = self._apply_operator_to_matrix(
                operator_adj_t,
                self.feature_preprojection_layer.weight.t(),
                x_steps=self._operator_steps(data),
            )
            if self.feature_preprojection_layer.bias is not None:
                x = x + self.feature_preprojection_layer.bias
            x = self.feature_preprojection_activation(x)
            x = self.feature_preprojection_dropout(x)
            return x
        x = self.feature_preprojection_layer(x)
        x = self.feature_preprojection_activation(x)
        x = self.feature_preprojection_dropout(x)
        return x

    def _forward_sparse_operator_sage_direct(self, data, *, gnn_adj_t):
        if not isinstance(self.gnn, GraphSAGE):
            raise ValueError(
                'lazy sparse operator direct mode currently supports model="sage". '
                'Use --feature-preprojection for other backbones.'
            )
        if not isinstance(gnn_adj_t, SparseTensor):
            raise TypeError('lazy sparse operator direct mode requires SparseTensor adjacency')

        operator_adj_t = self._operator_adj_t(
            data,
            device=self.gnn.conv1.lin_l.weight.device,
        )
        x_steps = self._operator_steps(data)

        conv1 = self.gnn.conv1
        neighbor_features = self._apply_operator_to_matrix(
            operator_adj_t,
            conv1.lin_l.weight.t(),
            x_steps=x_steps,
        )
        root_features = self._apply_operator_to_matrix(
            operator_adj_t,
            conv1.lin_r.weight.t(),
            x_steps=x_steps,
        )
        neighbor_features = self._apply_scale(neighbor_features)
        root_features = self._apply_scale(root_features)

        hidden = self._graphsage_mean_aggregate(gnn_adj_t, neighbor_features)
        if conv1.lin_l.bias is not None:
            hidden = hidden + conv1.lin_l.bias
        hidden = hidden + root_features
        if conv1.normalize:
            hidden = F.normalize(hidden, p=2.0, dim=-1)

        hidden = self.gnn.activation(hidden)
        hidden = self.gnn.dropout(hidden)
        return self.gnn.conv2(hidden, gnn_adj_t)

    def _refresh_smoother_cache(self, smoother_adj_t):
        if self.feature == 'operator':
            return
        smoother_adj_id = id(smoother_adj_t)
        if self._cached_smoother_adj_id is None:
            self._cached_smoother_adj_id = smoother_adj_id
            return
        if self._cached_smoother_adj_id != smoother_adj_id:
            self.smoother._cached_x = None
            self._cached_smoother_adj_id = smoother_adj_id

    @torch.no_grad()
    def refresh_smoother_cache(self, data, smoother_adj_t=None):
        if self.feature == 'operator':
            return self._build_feature_representation(data, smoother_adj_t)
        smoother_adj_t = data.adj_t if smoother_adj_t is None else smoother_adj_t
        smoother_adj_id = id(smoother_adj_t)

        self._cached_smoother_adj_id = smoother_adj_id
        self.smoother._cached_x = None
        return self._build_feature_representation(data, smoother_adj_t)

    def _build_feature_representation(self, data, smoother_adj_t):
        if self.feature == 'operator':
            if self._has_lazy_operator_input(data):
                if not self.feature_preprojection:
                    raise RuntimeError(
                        'lazy sparse operator direct mode is consumed via _forward_logits '
                        'and cannot be materialized as a dense feature matrix.'
                    )
                x = self._apply_operator_projection(None, data=data)
            else:
                x = self._apply_operator_projection(data.x)
            return self._apply_scale(x)

        x = self.smoother(data.x, smoother_adj_t)
        return self._apply_scale(x)

    def _forward_logits(self, data, gnn_adj_t=None, smoother_adj_t=None):
        smoother_adj_t = data.adj_t if smoother_adj_t is None else smoother_adj_t
        gnn_adj_t = data.adj_t if gnn_adj_t is None else gnn_adj_t

        if self.feature == 'operator' and self._has_lazy_operator_input(data) and not self.feature_preprojection:
            return self._forward_sparse_operator_sage_direct(data, gnn_adj_t=gnn_adj_t)

        self._refresh_smoother_cache(smoother_adj_t)

        x = self._build_feature_representation(data, smoother_adj_t)
        return self.gnn(x, gnn_adj_t)

    def forward(self, data, gnn_adj_t=None, smoother_adj_t=None):
        logits = self._forward_logits(data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)
        return F.softmax(logits, dim=1)

    @staticmethod
    def _target_indices(y):
        return y.argmax(dim=1) if y.dim() > 1 else y

    def training_step(self, data, gnn_adj_t=None, smoother_adj_t=None):
        logits = self._forward_logits(data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)
        target = self._target_indices(data.y)

        train_logits = logits[data.train_mask]
        train_target = target[data.train_mask]
        loss = F.cross_entropy(train_logits, train_target)

        metrics = {
            'train/loss': loss.detach(),
            'train/acc': self.accuracy(pred=train_logits, target=train_target) * 100,
        }

        return loss, metrics

    def validation_step(self, data, gnn_adj_t=None, smoother_adj_t=None):
        logits = self._forward_logits(data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)
        target = self._target_indices(data.y)

        metrics = {
            'val/loss': F.cross_entropy(logits[data.val_mask], target[data.val_mask]),
            'val/acc': self.accuracy(pred=logits[data.val_mask], target=target[data.val_mask]) * 100,
            'test/acc': self.accuracy(pred=logits[data.test_mask], target=target[data.test_mask]) * 100,
        }

        return metrics

    @staticmethod
    def accuracy(pred, target):
        pred = pred.argmax(dim=1) if len(pred.size()) > 1 else pred
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return accuracy_1d(pred=pred, target=target)

    @staticmethod
    def cross_entropy_loss(p_y, y, weighted=False):
        target_idx = y.argmax(dim=1) if y.dim() > 1 else y
        y_onehot = F.one_hot(target_idx, num_classes=p_y.size(1)).to(dtype=p_y.dtype)
        loss = -torch.log(p_y + 1e-20) * y_onehot
        loss *= y if weighted else 1
        loss = loss.sum(dim=1).mean()
        return loss
