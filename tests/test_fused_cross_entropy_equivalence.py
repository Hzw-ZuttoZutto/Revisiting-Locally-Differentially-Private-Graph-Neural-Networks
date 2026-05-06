import unittest

import torch
import torch.nn.functional as F


def manual_cross_entropy_from_logits(logits, target):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=logits.size(1)).to(dtype=logits.dtype)
    return (-(torch.log(probs + 1e-20) * target_onehot).sum(dim=1)).mean()


class FusedCrossEntropyEquivalenceTests(unittest.TestCase):
    def test_loss_and_gradient_match_manual_formula(self):
        torch.manual_seed(0)
        cases = [
            (8, 3),
            (64, 7),
            (256, 16),
        ]

        for batch_size, num_classes in cases:
            with self.subTest(batch_size=batch_size, num_classes=num_classes):
                logits_a = torch.randn(batch_size, num_classes, dtype=torch.float64, requires_grad=True)
                logits_b = logits_a.detach().clone().requires_grad_(True)
                target = torch.randint(num_classes, (batch_size,))

                loss_manual = manual_cross_entropy_from_logits(logits_a, target)
                loss_fused = F.cross_entropy(logits_b, target)

                loss_manual.backward()
                loss_fused.backward()

                self.assertTrue(torch.allclose(loss_manual, loss_fused, atol=1e-12, rtol=1e-10))
                self.assertTrue(torch.allclose(logits_a.grad, logits_b.grad, atol=1e-12, rtol=1e-10))


if __name__ == '__main__':
    unittest.main()
