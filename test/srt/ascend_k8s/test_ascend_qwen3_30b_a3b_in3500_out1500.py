import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_30B_A3B_MODEL_PATH,
    QWEN3_30B_A3B_OTHER_ARGS,
    QWEN3_30B_A3B_ENVS,
)


class TestQwen3_30B_A3B(TestSingleMixUtils):
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = QWEN3_30B_A3B_OTHER_ARGS
    envs = QWEN3_30B_A3B_ENVS
    max_out_len = 1500
    batch_size = 48
    num_prompts = int(batch_size) * 4
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_30b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
