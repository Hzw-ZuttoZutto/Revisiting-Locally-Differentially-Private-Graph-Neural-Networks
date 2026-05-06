import unittest
from unittest.mock import patch

import torch

from trainer import Trainer


class _DummyData:
    def __init__(self):
        self.x = torch.zeros(1, 1, dtype=torch.float32)
        self.adj_t = None

    def to(self, device):
        self.x = self.x.to(device)
        return self


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def training_step(self, data, gnn_adj_t=None, smoother_adj_t=None):
        _ = data, gnn_adj_t, smoother_adj_t
        loss = (self.w - 1.0).pow(2)
        metrics = {
            'train/loss': loss.detach(),
            'train/acc': torch.tensor(50.0, dtype=loss.dtype, device=loss.device),
        }
        return loss, metrics

    def validation_step(self, data, gnn_adj_t=None, smoother_adj_t=None):
        _ = data, gnn_adj_t, smoother_adj_t
        val_loss = (self.w - 1.0).abs().detach()
        return {
            'val/loss': val_loss,
            'val/acc': torch.tensor(60.0, dtype=val_loss.dtype, device=val_loss.device),
            'test/acc': torch.tensor(70.0, dtype=val_loss.dtype, device=val_loss.device),
        }


class _DummyLogger:
    def __init__(self):
        self.logged_metrics = []
        self.summary = None

    def log(self, metrics):
        self.logged_metrics.append(metrics)

    def log_summary(self, metrics):
        self.summary = metrics


class TrainerRuntimePolicyTests(unittest.TestCase):
    def test_log_every_epoch_false_only_logs_summary(self):
        logger = _DummyLogger()
        trainer = Trainer(
            max_epochs=3,
            learning_rate=0.01,
            patience=0,
            device='cpu',
            show_progress=False,
            log_every_epoch=False,
            logger=logger,
        )

        trainer.fit(_DummyModel(), _DummyData())

        self.assertEqual(len(logger.logged_metrics), 0)
        self.assertIsNotNone(logger.summary)
        self.assertIsInstance(logger.summary['train/loss'], float)
        self.assertIsInstance(logger.summary['val/loss'], float)
        self.assertIsInstance(logger.summary['test/acc'], float)

    def test_log_every_epoch_true_logs_each_epoch(self):
        logger = _DummyLogger()
        trainer = Trainer(
            max_epochs=4,
            learning_rate=0.01,
            patience=0,
            device='cpu',
            show_progress=False,
            log_every_epoch=True,
            logger=logger,
        )

        trainer.fit(_DummyModel(), _DummyData())

        self.assertEqual(len(logger.logged_metrics), 4)
        for metrics in logger.logged_metrics:
            self.assertIsInstance(metrics['train/loss'], float)
            self.assertIsInstance(metrics['val/loss'], float)
            self.assertIsInstance(metrics['test/acc'], float)
        self.assertIsNotNone(logger.summary)

    def test_show_progress_false_does_not_invoke_tqdm(self):
        trainer = Trainer(
            max_epochs=2,
            learning_rate=0.01,
            patience=0,
            device='cpu',
            show_progress=False,
            log_every_epoch=False,
            logger=None,
        )

        with patch('trainer.tqdm', side_effect=AssertionError('tqdm should not be called')):
            trainer.fit(_DummyModel(), _DummyData())

    def test_show_progress_true_invokes_tqdm(self):
        trainer = Trainer(
            max_epochs=2,
            learning_rate=0.01,
            patience=0,
            device='cpu',
            show_progress=True,
            log_every_epoch=False,
            logger=None,
        )
        marker = {'called': False}

        class _FakeProgress:
            def __init__(self, iterable):
                self._iterable = iterable

            def __iter__(self):
                return iter(self._iterable)

            def set_postfix(self, *args, **kwargs):
                _ = args, kwargs
                return None

        def _fake_tqdm(iterable, **kwargs):
            _ = kwargs
            marker['called'] = True
            return _FakeProgress(iterable)

        with patch('trainer.tqdm', side_effect=_fake_tqdm):
            trainer.fit(_DummyModel(), _DummyData())

        self.assertTrue(marker['called'])


if __name__ == '__main__':
    unittest.main()
