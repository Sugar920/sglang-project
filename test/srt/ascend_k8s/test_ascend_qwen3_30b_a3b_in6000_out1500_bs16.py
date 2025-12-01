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
    dataset_name = "random"
    request_rate = 5.5
    max_concurrency = 16
    input_len = 6000
    output_len = 1500
    random_range_ratio = 0.5
    ttft = 2760.9
    tpot = 24.95
    output_token_throughput = 597.42

    def test_qwen3_30b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
