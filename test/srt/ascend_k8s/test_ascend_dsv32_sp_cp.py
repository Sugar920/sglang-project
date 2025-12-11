import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from test_ascend_disaggregation_utils import (
    TestAscendDisaggregationUtils,
)

MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "2800",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_USE_AG_AFTER_QLORA": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_MOE_NZ": "1",
        "PROFILING_MODE": "dynamic",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": "enp23s0f3",
        "GLOO_SOCKET_IFNAME": "enp23s0f3",    
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFIMITY": "1",
        "LD_LIBRARY_PATH": "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}",
        "ASCEND_HOME_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "1024",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_USE_AG_AFTER_QLORA": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_FUSED_MOE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "TASK_QUEUE_ENABLE": "0",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "HCCL_SOCKET_IFNAME": "enp23s0f3",
        "GLOO_SOCKET_IFNAME": "enp23s0f3",     
    },
    "prefill_args": [
        "--nnodes",
        2,
        "--disaggregation-mode",
        "prefill",
        "--enable-sp",
        "--tp",
        32,
        "--cp-size",
        32,
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.85,
        "--max-total-tokens",
        68000,
        "--context-length",
        68000,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        327680,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        "16",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--quantization",
        "w8a8_int8",
        "--disable-cuda-graph",      
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--enable-sp",
        "--tp",
        32,
        "--dp",
        4,
        "--ep",
        32,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.85,
        "--max-total-tokens",
        68000,
        "--context-length",
        68000,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        327680,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        "128",
        "--cuda-graph-max-bs",
        "32",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "w8a8_int8",
        "--prefill-round-robin-balance",  
        "--load-balance-method",
        "round_robin",
    ],
}




TEST_MODEL_MATRIX = {
    "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8": {
        "accuracy": 0.95,
        "latency": 1000,
    },
}


class TestAscendEnableMixedChunk(TestAscendDisaggregationUtils):
    def test_a_gsm8k(self):
        with self.subTest(model=MODEL_PATH):
            print(f"##=== Testing accuracy: {model} ===##")
            try:
                args = SimpleNamespace(
                    num_shots=20,
                    data_path=None,
                    num_questions=200,
                    max_new_tokens=512,
                    parallel=64,
                    host=f"http://192.168.0.60",
                    port=6688,
                )

                metrics = run_eval_few_shot_gsm8k(args)
#                self.assertGreaterEqual(
 #                   metrics["accuracy"],
  #                  TEST_MODEL_MATRIX[model]["accuracy"],
   #             )
            finally:
                pass


if __name__ == "__main__":
    unittest.main()
