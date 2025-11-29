import unittest

from test_ascend_disaggregation_utils import (
    TestAscendDisaggregationUtils,
)

MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8"

MODEL_CONFIG = {
    "model_path": MODEL_PATH,
    "prefill_envs": {
        "SGLANG_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        "SGLANG_USE_AG_AFTER_QLORA": "1",
        "HCCL_BUFFSIZE": "1536",
        "TASK_QUEUE_ENABLE": "2",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
    },
    "decode_envs": {
        "SGLANG_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        # "ENABLE_MOE_NZ": "1",
        # "NO_DP_ALL_GATHER": "1",
        # "ENABLE_FUSED_MOE": "1",
        "DP_ROUND_ROBIN": "1",
        "HCCL_BUFFSIZE": "1200",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        # "HCCL_SOCKET_IFNAME": "data0.3001",
        # "GLOO_SOCKET_IFNAME": "data0.3001",
    },
    "prefill_args": [
        "--quantization",
        "w8a8_int8",
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        1,
        "--node-rank",
        0,
        "--tp-size",
        16,
        "--dp-size",
        2,
        "--mem-fraction-static",
        0.81,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        32768,
        "--max-prefill-tokens",
        28680,
        "--max-running-requests",
        8,
        "--context-length",
        8192,
        "--disable-overlap-schedule",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/home/fuyong/codes/sglang/sgl_ascend/hot_map/aisbench_hot_map.pt",
        "--enable-dp-attention",
        # "--tokenizer-worker-num",
        # 4,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        # "--enable-expert-distribution-metrics",
        "--disable-shared-experts-fusion",
        "--disable-cuda-graph",
    ],
    "decode_args": [
        "--quantization",
        "w8a8_int8",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        32,
        "--dp-size",
        32,
        "--mem-fraction-static",
        0.9,
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        1,
        "--disable-cuda-graph",
        "--watchdog-timeout",
        9000,
        "--context-length",
        8192,
        "--max-running-requests",
        768,
        "--prefill-round-robin-balance",
        "--cuda-graph-bs",
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        # "--ep-dispatch-algorithm",
        # "static",
        # "--init-expert-location",
        # "/data/.cache/hot_map/aisbench_hot_map_decode.pt",
        # "--ep-num-redundant-experts",
        # 64,
        "--tokenizer-worker-num",
        4,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        1,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        2,
        # "--enable-expert-distribution-metrics",
        "--disable-shared-experts-fusion",
    ],
}


class Test_DeepSeek_R1_W4A8_2P1D(TestAscendDisaggregationUtils):
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = None
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = None
    output_len = None
    random_range_ratio = 0.5
    ttft = None
    tpot = None
    output_token_throughput = None

    def test_throughput(self):
        self.run_throughput()

if __name__ == "__main__":
    unittest.main()
