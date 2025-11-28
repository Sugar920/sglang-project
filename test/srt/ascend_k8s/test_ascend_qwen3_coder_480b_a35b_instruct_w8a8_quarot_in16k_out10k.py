import unittest

from test_ascend_single_mix_utils import (
    TestSingleMixUtils,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS,
    QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS,
)


class TestQwen3_Coder_480B_A35b_Instruct_W8a8_Quarot(TestSingleMixUtils):
    model = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_MODEL_PATH
    dataset = (
        "/data/ascend-ci-share-pkking-sglang/modelscope/hub/datasets/Qwen3-32B-w8a8-MindIE/GSM8K-in3500-bs5000/test.jsonl")
    other_args = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_OTHER_ARGS
    envs = QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_ENVS
    max_out_len = 1500
    batch_size = 48
    num_prompts = int(batch_size) * 4
    tpot = 100
    output_token_throughput = 300

    def test_qwen3_coder_480b_a35b_instruct_w8a8_quarot(self):
        self.run_ais_bench()


if __name__ == "__main__":
    unittest.main()
