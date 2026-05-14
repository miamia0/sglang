import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.test.test_utils import CustomTestCase


class TestDeepSeekV4CompressState(CustomTestCase):
    def test_pool_keeps_padding_state_row(self):
        pool = CompressStatePool(
            size=2,
            ring_size=1,
            overlap=True,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=4,
        )

        self.assertEqual(pool.kv_score_buffer.kv_score.shape, (4, 32))
        pool.kv_score_buffer[2].clear()


if __name__ == "__main__":
    unittest.main()
