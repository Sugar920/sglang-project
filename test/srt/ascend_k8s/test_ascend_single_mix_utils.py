import os
import subprocess

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

QWEN3_32B_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
QWEN3_235B_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
# DEEPSEEK_R1_0528_W4A8_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-R1-0528-w4a8"
DEEPSEEK_R1_0528_W4A8_MODEL_PATH = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
QWEN3_32B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "w8a8_int8",
        "--max-running-requests",
        "78",
        "--context-length",
        "8192",
        "--enable-hierarchical-cache",
        "--hicache-write-policy",
        "write_through",
        "--hicache-ratio",
        "3",
        "--chunked-prefill-size",
        "43008",
        "--max-prefill-tokens",
        "52500",
        "--tp-size",
        "4",
        "--mem-fraction-static",
        "0.68",
        "--cuda-graph-bs",
        "78",
        "--dtype",
        "bfloat16"
    ]
    if is_npu()
    else []
)

QWEN3_32B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "400",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_235B_OTHER_ARGS = (
    [
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "w8a8_int8",
        "--max-running-requests",
        "576",
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "102400",
        "--max-prefill-tokens",
        "458880",
        "--disable-radix-cache",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--tp-size",
        "16",
        "--dp-size",
        "16",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--mem-fraction-static",
        "0.8",
        "--cuda-graph-bs",
        6,
        12,
        18,
        36,
    ]
    if is_npu()
    else []
)

QWEN3_235B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2100",
    "HCCL_SOCKET_IFNAME": "DATA0.3001",
    "GLOO_SOCKET_IFNAME": "DATA0.3001",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ENABLE_ASCEND_MOE_NZ": "1",
}

DEEPSEEK_R1_0528_W4A8_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "1600",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
}

DEEPSEEK_R1_0528_W4A8_OTHER_ARGS = (
    [
        "--tp",
        "16",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "w8a8_int8",
        "--watchdog-timeout",
        "9000",
        "--cuda-graph-bs",
        "8",
        # "16",
        # "24",
        # "28",
        # "32",
        "--mem-fraction-static",
        # "0.68",
        "0.8",
        "--max-running-requests",
        # "128",
        "8",
        "--context-length",
        "8188",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "6000",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--enable-dp-attention",
        "--dp-size",
        "4",
        "--enable-dp-lm-head",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--dtype",
        "bfloat16",
    ]
)

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None


class TestSingleMixUtils(CustomTestCase):
    model = None
    dataset = None
    other_args = None
    envs = None
    max_out_len = None
    batch_size = 78
    num_prompts = int(batch_size) * 4
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        if is_npu():
            env = os.environ.copy()
            env.update(cls.envs)
        else:
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_gsm8k(self):
        port = self.base_url.split(":")[-1]
        run_command("rm -rf ./benchmark")
        run_command("pip3 install nltk==3.8")
        run_command("git clone https://gitee.com/aisbench/benchmark.git")
        run_command(
            f'sed -i \'s#path="[^"]*"#path="{self.model}"#\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            'sed -i \'s/model="[^"]*"/model="Qwen3"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            "sed -i 's/request_rate = [^\"]*/request_rate = 5.5,/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            'sed -i \'s/host_ip = "[^"]*"/host_ip = "127.0.0.1"/\' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py'
        )
        run_command(
            f"sed -i 's/host_port = [^\"]*/host_port = {port},/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            f"sed -i 's/max_out_len = [^\"]*/max_out_len = {self.max_out_len},/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            f"sed -i 's/batch_size=[^\"]*/batch_size={self.batch_size},/' ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        run_command(
            r"""sed -i '/generation_kwargs = dict(/,/),/c\        generation_kwargs = dict(\n            temperature = 0,\n            ignore_eos = True,\n        ),'  ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"""
        )
        run_command("mkdir ./benchmark/ais_bench/datasets/gsm8k")
        run_command(f"\cp {self.dataset} ./benchmark/ais_bench/datasets/gsm8k/")
        run_command("touch ./benchmark/ais_bench/datasets/gsm8k/train.jsonl")
        ais_res = run_command("pip3 install -e ./benchmark/")
        print(str(ais_res))
        cat_res = run_command(
            "cat ./benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py"
        )
        print("cat_res is " + str(cat_res))
        metrics = run_command(
            f"ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts {self.num_prompts} | tee ./gsm8k_deepseek_log.txt"
        )
        print("metrics is " + str(metrics))
        res_ttft = run_command(
            "cat ./gsm8k_deepseek_log.txt | grep TTFT | awk '{print $6}'"
        )
        res_tpot = run_command(
            "cat ./gsm8k_deepseek_log.txt | grep TPOT | awk '{print $6}'"
        )
        res_output_token_throughput = run_command(
            "cat ./gsm8k_deepseek_log.txt | grep 'Output Token Throughput' | awk '{print $8}'"
        )
        self.assertLessEqual(
            float(res_tpot),
            self.tpot,
        )
        self.assertGreaterEqual(
            float(res_output_token_throughput),
            self.output_token_throughput,
        )
