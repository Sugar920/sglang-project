import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_8B_MODEL_PATH,
    QWEN3_8B_OTHER_ARGS,
    QWEN3_8B_ENVS,
)


class TestQWEN3_8B(TestSingleMixUtils):
    model = QWEN3_8B_MODEL_PATH
    dataset = (
        "/data/ascend-ci-share-pkking-sglang/modelscope/hub/datasets/Qwen3-32B-w8a8-MindIE/GSM8K-in3500-bs5000/test.jsonl")
    other_args = QWEN3_8B_OTHER_ARGS
    envs = QWEN3_8B_ENVS
    max_out_len = 300
    batch_size = 16
    num_prompts = int(batch_size) * 4
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_8b(self):
        self.run_ais_bench()


if __name__ == "__main__":
    unittest.main()
