import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_30B_A3B_MODEL_PATH,
    QWEN3_30B_A3B_OTHER_ARGS,
    QWEN3_30B_A3B_ENVS,
)


class TestQwen3_30B_A3B(TestSingleMixUtils):
    model = QWEN3_30B_A3B_MODEL_PATH
    dataset = (
        "/data/ascend-ci-share-pkking-sglang/modelscope/hub/datasets/Qwen3-32B-w8a8-MindIE/GSM8K-in3500-bs5000/test.jsonl")
    other_args = QWEN3_30B_A3B_OTHER_ARGS
    envs = QWEN3_30B_A3B_ENVS
    max_out_len = 1500
    batch_size = 48
    num_prompts = int(batch_size) * 4
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_30b_a3b(self):
        self.run_ais_bench()


if __name__ == "__main__":
    unittest.main()
