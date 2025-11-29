import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    DEEPSEEK_R1_0528_W4A8_MODEL_PATH,
    DEEPSEEK_R1_0528_W4A8_OTHER_ARGS,
    DEEPSEEK_R1_0528_W4A8_ENVS,
)


class TestDEEPSEEK_R1_0528_W4A8(TestSingleMixUtils):
    model = DEEPSEEK_R1_0528_W4A8_MODEL_PATH
    dataset = (
        "/data/ascend-ci-share-pkking-sglang/modelscope/hub/datasets/DeepSeek-R1-0528-w4a8/GSM8K-in3500-bs5000/test.jsonl")
    other_args = DEEPSEEK_R1_0528_W4A8_OTHER_ARGS
    envs = DEEPSEEK_R1_0528_W4A8_ENVS
    max_out_len = 1500
    # batch_size = 128
    batch_size = 8
    num_prompts = int(batch_size) * 4
    ttft = 5000
    tpot = 50
    output_token_throughput = 300

    def test_deepseek_r1_0528_w4a8(self):
        self.run_ais_bench()


if __name__ == "__main__":
    unittest.main()
