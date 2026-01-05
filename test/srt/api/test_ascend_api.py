import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAscendApi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        
    def test_api_health(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health")
        self.assertEqual(response.status_code, 200)

    def test_api_health_generate(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

    def test_api_ping(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/ping")
        self.assertEqual(response.status_code, 200)

    def test_api_model_info(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/model_info")
        print(f"[test_api_model_info] response.text: {response.text}")
        print(response.text["model_path"])
        print(f"[test_api_model_info] response.json(): {response.json()}")
        print(response.json()['model_path'])
        # model_path = response.json()["model_path"]
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(model_path, self.model)

    def test_api_weight_version(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/weight_version")
        print(f"[test_api_model_info] response: {response}")

    def test_api_server_info(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        print(f"[test_api_model_info] response: {response}")

    def test_api_get_load(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_load")
        print(f"[test_api_model_info] response: {response}")

    def test_api_v1_models(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/v1/models")
        print(f"[test_api_model_info] response: {response}")

    def test_api_v1_models_path(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/v1/models{model:path}")
        print(f"[test_api_model_info] response: {response}")
        
        # response = requests.post(
        #     f"{DEFAULT_URL_FOR_TEST}/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": 32,
        #         },
        #     },
        # )
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        # response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(response.json()["enable_mixed_chunk"], True)
        # kill_process_tree(process.pid)


if __name__ == "__main__":

    unittest.main()
