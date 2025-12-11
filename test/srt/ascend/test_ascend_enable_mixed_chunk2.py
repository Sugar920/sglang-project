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

TEST_MODEL_MATRIX = {
    "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B": {
        "accuracy": 0.95,
        "latency": 1000,
    },
}


class TestAscendEnableMixedChunk(CustomTestCase):
    metrics_with_mixed_chunk = None
    metrics_without_mixed_chunk = None

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.common_args_with_mixed_chunk = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
            "--tp-size",
            "4",
            "--enable-mixed-chunk",
            "--max-running-requests",
            4,
            "--cuda-graph-bs",
            1,
            2,
            3,
            4,
            "--disable-radix-cache",
        ]

        cls.common_args_without_mixed_chunk = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
            "--tp-size",
            "4",
            "--max-running-requests",
            4,
            "--cuda-graph-bs",
            1,
            2,
            3,
            4,
            "--disable-radix-cache",
        ]

        cls.extra_envs = {
            "SGLANG_SET_CPU_AFFINITY": "1",
            "STREAMS_PER_DEVICE": "32",
            "HCCL_BUFFSIZE": "1536",
            "HCCL_OP_EXPANSION_MODE": "AIV",
        }
        os.environ.update(cls.extra_envs)

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
                print(f"##=== Testing accuracy: {model} ===##")

                process_with_mixed_chunk = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1500,
                    other_args=[
                        *self.common_args_with_mixed_chunk,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=5,
                        data_path=None,
                        num_questions=1319,
                        max_new_tokens=512,
                        parallel=256,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    self.metrics_with_mixed_chunk = run_eval_few_shot_gsm8k(args)
                finally:
                    kill_process_tree(process_with_mixed_chunk.pid)

                process_without_mixed_chunk = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=1500,
                    other_args=[
                        *self.common_args_without_mixed_chunk,
                    ],
                )

                try:
                    args = SimpleNamespace(
                        num_shots=5,
                        data_path=None,
                        num_questions=1319,
                        max_new_tokens=512,
                        parallel=256,
                        host=f"http://{self.url.hostname}",
                        port=int(self.url.port),
                    )

                    self.metrics_without_mixed_chunk = run_eval_few_shot_gsm8k(args)
                    accuracy_with_mixed_chunk = self.metrics_with_mixed_chunk["accuracy"]
                    accuracy_without_mixed_chunk = self.metrics_without_mixed_chunk["accuracy"]
                    if accuracy_with_mixed_chunk < accuracy_without_mixed_chunk:
                        accuracy_diff = (accuracy_without_mixed_chunk - accuracy_with_mixed_chunk) / accuracy_without_mixed_chunk
                        self.assertLessEqual(
                           accuracy_diff,
                           0.01,
                       )
                finally:
                    kill_process_tree(process_without_mixed_chunk.pid)


if __name__ == "__main__":
    unittest.main()
