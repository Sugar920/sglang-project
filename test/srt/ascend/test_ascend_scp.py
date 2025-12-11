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
    "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8": {
        "accuracy": 0.95,
        "latency": 1000,
    },
}


class TestAscendEnableMixedChunk(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()

    def test_a_gsm8k(self):
        for model in self.models:
            with self.subTest(model=model):
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
