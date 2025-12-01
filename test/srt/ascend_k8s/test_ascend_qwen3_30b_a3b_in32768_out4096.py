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
    max_concurrency = 48
    input_len = 32768
    output_len = 4096
    random_range_ratio = 0.5
    ttft = 19656.76
    tpot = 48.32
    output_token_throughput = 298.89*0.5

    def test_qwen3_30b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
