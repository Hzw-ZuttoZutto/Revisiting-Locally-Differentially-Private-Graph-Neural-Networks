import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from pre_smoothing_feature_cache import (
    CACHE_PAYLOAD_SCHEMA_VERSION,
    DENSE_TENSOR_FORMAT,
    SPARSE_FLAT_TENSOR_FORMAT,
    _atomic_torch_save,
    _decode_tensor_from_cache,
    _encode_tensor_for_cache,
    _load_payload,
    _restore_from_payload,
    compact_legacy_cache_file,
    estimate_adaptive_tensor_storage_bytes,
)


def _payload_for(encoded):
    return {
        "schema_version": CACHE_PAYLOAD_SCHEMA_VERSION,
        "mechanism_output_range_before_nfr": 3.5,
        "resolved_m": 2,
        "mechanism": "mbm",
        **encoded,
    }


class CompactPreSmoothingCacheTest(unittest.TestCase):
    def test_sparse_round_trip_is_exact(self):
        original = torch.zeros((7, 11), dtype=torch.float32)
        original[0, 1] = 1.25
        original[3, 5] = -9.5
        original[6, 10] = 2.0

        encoded, storage_bytes = _encode_tensor_for_cache(original)
        self.assertEqual(encoded["x_format"], SPARSE_FLAT_TENSOR_FORMAT)
        self.assertLess(storage_bytes, original.numel() * original.element_size())

        restored = _decode_tensor_from_cache(
            _payload_for(encoded),
            device=torch.device("cpu"),
            dtype=original.dtype,
        )
        self.assertTrue(torch.equal(restored, original))

    def test_dense_tensor_uses_dense_fallback(self):
        original = torch.arange(1, 65, dtype=torch.float32).view(8, 8)
        encoded, storage_bytes = _encode_tensor_for_cache(original)
        self.assertEqual(encoded["x_format"], DENSE_TENSOR_FORMAT)
        self.assertEqual(storage_bytes, original.numel() * original.element_size())
        restored = _decode_tensor_from_cache(
            _payload_for(encoded),
            device=torch.device("cpu"),
            dtype=original.dtype,
        )
        self.assertTrue(torch.equal(restored, original))

    def test_legacy_dense_payload_remains_readable(self):
        original = torch.tensor([[0.0, 2.0], [-3.0, 0.0]], dtype=torch.float32)
        payload = {
            "schema_version": 1,
            "x": original.clone(),
            "mechanism_output_range_before_nfr": 4.0,
            "resolved_m": 1,
            "mechanism": "pm",
        }
        data = SimpleNamespace(x=torch.empty_like(original))
        result = _restore_from_payload(data, payload, Path("legacy.pt"))
        self.assertTrue(torch.equal(data.x, original))
        self.assertEqual(data.output_range, 4.0)
        self.assertEqual(data.feature_mechanism_resolved_m, 1)
        self.assertEqual(result.bytes_written, original.numel() * original.element_size())

    def test_atomic_file_round_trip_accepts_compact_payload(self):
        original = torch.zeros((32, 64), dtype=torch.float32)
        original[:, 3] = torch.arange(32, dtype=torch.float32)
        encoded, _ = _encode_tensor_for_cache(original)
        payload = _payload_for(encoded)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.pt"
            _atomic_torch_save(path, payload)
            loaded = _load_payload(path)
            self.assertIsNotNone(loaded)
            restored = _decode_tensor_from_cache(
                loaded,
                device=torch.device("cpu"),
                dtype=original.dtype,
            )
        self.assertTrue(torch.equal(restored, original))

    def test_legacy_file_can_be_compacted_in_place(self):
        original = torch.zeros((128, 256), dtype=torch.float32)
        original[:, 7] = torch.linspace(-3.0, 3.0, steps=128)
        legacy = {
            "schema_version": 1,
            "x": original.clone(),
            "mechanism_output_range_before_nfr": 8.0,
            "resolved_m": 1,
            "mechanism": "hds",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            _atomic_torch_save(path, legacy)
            status, before, after = compact_legacy_cache_file(path)
            self.assertEqual(status, "compacted")
            self.assertLess(after, before)
            loaded = _load_payload(path)
            self.assertEqual(loaded["schema_version"], CACHE_PAYLOAD_SCHEMA_VERSION)
            restored = _decode_tensor_from_cache(
                loaded,
                device=torch.device("cpu"),
                dtype=original.dtype,
            )
            second_status, _, _ = compact_legacy_cache_file(path)
        self.assertTrue(torch.equal(restored, original))
        self.assertEqual(second_status, "already_compact")

    def test_storage_estimate_is_conservative_for_sparse_rows(self):
        shape = (7575, 12047)
        estimated = estimate_adaptive_tensor_storage_bytes(
            shape,
            element_size=4,
            max_nonzero=shape[0] * 4,
        )
        self.assertEqual(estimated, shape[0] * 4 * (4 + 4))
        self.assertLess(estimated, shape[0] * shape[1] * 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU restoration test")
    def test_sparse_round_trip_on_gpu_is_exact(self):
        original = torch.zeros((128, 256), dtype=torch.float32)
        original[:, 17] = torch.linspace(-2.0, 2.0, steps=128)
        encoded, _ = _encode_tensor_for_cache(original.cuda())
        restored = _decode_tensor_from_cache(
            _payload_for(encoded),
            device=torch.device("cuda:0"),
            dtype=original.dtype,
        )
        self.assertTrue(torch.equal(restored.cpu(), original))


if __name__ == "__main__":
    unittest.main()
