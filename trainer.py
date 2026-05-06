import csv
import os
import sys
import torch
from torch.optim import SGD, Adam
from tqdm.auto import tqdm


class Trainer:
    def __init__(
            self,
            optimizer:      dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
            max_epochs:     dict(help='maximum number of training epochs') = 500,
            learning_rate:  dict(help='learning rate') = 0.01,
            weight_decay:   dict(help='weight decay (L2 penalty)') = 0.0,
            patience:       dict(help='early-stopping patience window size') = 0,
            collect_grad_stats: dict(help='collect loss/gradient diagnostics per epoch and export plots') = False,
            gradient_clip: dict(help='enable gradient clipping via torch.nn.utils.clip_grad_norm_') = False,
            gradient_clip_max_norm: dict(help='max norm for gradient clipping (active only with --gradient-clip)', type=float) = 1.0,
            sim_epoch_refresh: dict(help='re-sample sim artificial features after each epoch and precompute feature smoother output for the next epoch') = False,
            show_progress: dict(help='show training progress bars and per-epoch postfix metrics') = False,
            log_every_epoch: dict(help='when --log=true, emit per-epoch metrics to wandb (otherwise summary only)') = False,
            device='cuda',
            logger=None,
    ):
        if gradient_clip_max_norm <= 0:
            raise ValueError('gradient_clip_max_norm must be > 0')

        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.collect_grad_stats = collect_grad_stats
        self.gradient_clip = gradient_clip
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.sim_epoch_refresh = sim_epoch_refresh
        self.show_progress = bool(show_progress)
        self.log_every_epoch = bool(log_every_epoch)
        self.logger = logger
        self.model = None

    def configure_optimizers(self):
        if self.optimizer_name == 'sgd':
            return SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(
            self,
            model,
            data,
            eval_data=None,
            train_gnn_adj_provider=None,
            train_smoother_adj_t=None,
            eval_gnn_adj_t=None,
            eval_smoother_adj_t=None,
            diagnostic_dir=None,
            epoch_end_data_refresh_fn=None,
    ):
        self.model = model.to(self.device)
        data = data.to(self.device)
        eval_data = data if eval_data is None else eval_data.to(self.device)

        if hasattr(self.model, 'clear_cached_state'):
            self.model.clear_cached_state()
        if self.sim_epoch_refresh and epoch_end_data_refresh_fn is None:
            raise ValueError('sim_epoch_refresh requires epoch_end_data_refresh_fn to be provided.')
        if self.sim_epoch_refresh and not hasattr(self.model, 'refresh_smoother_cache'):
            raise ValueError('sim_epoch_refresh requires the model to implement refresh_smoother_cache().')

        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_metrics = None

        training_trace = [] if self.collect_grad_stats else None
        epoch_iterator = range(1, self.max_epochs + 1)
        epoch_progbar = None
        if self.show_progress:
            epoch_progbar = tqdm(epoch_iterator, desc='Epoch: ', leave=False, position=1, file=sys.stdout)
            epoch_iterator = epoch_progbar

        for epoch in epoch_iterator:
            metrics = {'epoch': epoch}
            current_train_gnn_adj_t = train_gnn_adj_provider(epoch) if train_gnn_adj_provider is not None else None
            train_metrics = self._train(
                data,
                optimizer,
                gnn_adj_t=current_train_gnn_adj_t,
                smoother_adj_t=train_smoother_adj_t,
            )
            metrics.update(train_metrics)

            current_eval_gnn_adj_t = eval_data.adj_t if eval_gnn_adj_t is None else eval_gnn_adj_t
            val_metrics = self._validation(
                eval_data,
                gnn_adj_t=current_eval_gnn_adj_t,
                smoother_adj_t=eval_smoother_adj_t,
            )
            metrics.update(val_metrics)

            if self.collect_grad_stats:
                training_trace.append({
                    'epoch': int(epoch),
                    'train/loss': self._metric_to_float(metrics['train/loss']),
                    'val/loss': self._metric_to_float(metrics['val/loss']),
                    'train/grad_norm_global_preclip': self._metric_to_float(
                        metrics.get('train/grad_norm_global_preclip', 0.0)
                    ),
                    'train/grad_norm_conv1_preclip': self._metric_to_float(
                        metrics.get('train/grad_norm_conv1_preclip', 0.0)
                    ),
                    'train/grad_norm_conv2_preclip': self._metric_to_float(
                        metrics.get('train/grad_norm_conv2_preclip', 0.0)
                    ),
                })

            if self.logger and self.log_every_epoch:
                self.logger.log(self._metrics_to_scalars(metrics))

            val_loss = self._metric_to_float(metrics['val/loss'])
            val_acc = self._metric_to_float(metrics['val/acc'])
            train_acc = self._metric_to_float(metrics['train/acc'])

            should_stop = False
            if best_metrics is None or (
                val_loss < self._metric_to_float(best_metrics['val/loss']) and
                self._metric_to_float(best_metrics['val/acc']) < val_acc <= 100.0 and
                self._metric_to_float(best_metrics['train/acc']) < train_acc <= 105.0
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    should_stop = True

            # display metrics on progress bar
            if epoch_progbar is not None:
                epoch_progbar.set_postfix(self._metrics_to_scalars(metrics))

            if should_stop:
                break

            has_next_epoch = epoch < self.max_epochs
            if self.sim_epoch_refresh and has_next_epoch and epoch_end_data_refresh_fn is not None:
                epoch_end_data_refresh_fn(epoch=epoch, data=data, eval_data=eval_data)
                current_train_smoother_adj_t = data.adj_t if train_smoother_adj_t is None else train_smoother_adj_t
                self.model.refresh_smoother_cache(data, smoother_adj_t=current_train_smoother_adj_t)

        if self.logger and best_metrics is not None:
            self.logger.log_summary(self._metrics_to_scalars(best_metrics))

        if self.collect_grad_stats:
            self._save_training_diagnostics(training_trace=training_trace, diagnostic_dir=diagnostic_dir)

        return best_metrics

    def _train(self, data, optimizer, gnn_adj_t=None, smoother_adj_t=None):
        self.model.train()
        optimizer.zero_grad()
        loss, metrics = self.model.training_step(data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)
        loss.backward()

        preclip_global_grad_norm = None
        preclip_conv1_grad_norm = None
        preclip_conv2_grad_norm = None
        if self.collect_grad_stats or self.gradient_clip:
            preclip_global_grad_norm = self._module_grad_norm(self.model)
            gnn_module = getattr(self.model, 'gnn', None)
            preclip_conv1_grad_norm = self._module_grad_norm(getattr(gnn_module, 'conv1', None))
            preclip_conv2_grad_norm = self._module_grad_norm(getattr(gnn_module, 'conv2', None))

        if self.gradient_clip:
            params = [param for param in self.model.parameters() if param.requires_grad]
            if len(params) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.gradient_clip_max_norm)

        optimizer.step()

        if self.collect_grad_stats:
            metrics.update({
                'train/grad_norm_global_preclip': preclip_global_grad_norm,
                'train/grad_norm_conv1_preclip': preclip_conv1_grad_norm,
                'train/grad_norm_conv2_preclip': preclip_conv2_grad_norm,
            })

        return metrics

    @torch.no_grad()
    def _validation(self, data, gnn_adj_t=None, smoother_adj_t=None):
        self.model.eval()
        return self.model.validation_step(data, gnn_adj_t=gnn_adj_t, smoother_adj_t=smoother_adj_t)

    @staticmethod
    def _module_grad_norm(module):
        if module is None:
            return 0.0

        total = None
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            grad_sq_sum = parameter.grad.detach().pow(2).sum()
            total = grad_sq_sum if total is None else total + grad_sq_sum

        if total is None:
            return 0.0

        return float(torch.sqrt(total).item())

    @staticmethod
    def _metric_to_float(value):
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    @classmethod
    def _metrics_to_scalars(cls, metrics):
        scalar_metrics = {}
        for key, value in metrics.items():
            if key == 'epoch':
                scalar_metrics[key] = int(value)
                continue
            if torch.is_tensor(value):
                scalar_metrics[key] = float(value.detach().item())
                continue
            scalar_metrics[key] = value
        return scalar_metrics

    @staticmethod
    def _prepare_diagnostic_dir(diagnostic_dir):
        if diagnostic_dir is None:
            return None
        os.makedirs(diagnostic_dir, exist_ok=True)
        return diagnostic_dir

    def _save_training_diagnostics(self, training_trace, diagnostic_dir):
        if training_trace is None or len(training_trace) == 0:
            return

        diagnostic_dir = self._prepare_diagnostic_dir(diagnostic_dir)
        if diagnostic_dir is None:
            return

        trace_file_path = os.path.join(diagnostic_dir, 'training_trace.csv')
        fieldnames = list(training_trace[0].keys())
        with open(trace_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_trace)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f'[Diagnostics] Skip plotting because matplotlib is unavailable: {exc}')
            return

        epochs = [row['epoch'] for row in training_trace]
        train_loss = [row['train/loss'] for row in training_trace]
        val_loss = [row['val/loss'] for row in training_trace]
        grad_norm_global = [row['train/grad_norm_global_preclip'] for row in training_trace]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, train_loss, label='train/loss')
        ax.plot(epochs, val_loss, label='val/loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Loss Curve')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(diagnostic_dir, 'loss_curve.png'), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, grad_norm_global, label='global pre-clip grad norm')
        ax.set_xlabel('epoch')
        ax.set_ylabel('gradient norm')
        ax.set_title('Gradient Norm Curve')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(diagnostic_dir, 'grad_norm_curve.png'), dpi=200)
        plt.close(fig)
