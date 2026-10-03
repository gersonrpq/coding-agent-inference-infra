"""Values that live in several files must say the same thing: the output cap and the context window."""
import json
import os
import re
import unittest

import yaml

from tests._env import ROOT, guard


def read(path):
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


class OutputCapAndContext(unittest.TestCase):
    def test_litellm_model_info_matches_the_guard_cap(self):
        caps = {int(m) for m in re.findall(r"max_output_tokens:\s*(\d+)", read("cluster/litellm/config.yaml"))}
        self.assertEqual(caps, {guard.MAX_OUTPUT_TOKENS})

    def test_pi_asks_for_exactly_the_cap(self):
        models = json.loads(read("app/pi/agent/models.json"))
        found = [m["maxTokens"] for p in models["providers"].values() for m in p["models"]]
        self.assertTrue(found)
        self.assertEqual(set(found), {guard.MAX_OUTPUT_TOKENS})

    def test_context_window_is_the_engines_everywhere(self):
        engine = int(yaml.safe_load(read("cluster/sglang/config.yaml"))["data"]["MAX_MODEL_LEN"])
        config = read("cluster/litellm/config.yaml")
        self.assertEqual({int(m) for m in re.findall(r"max_input_tokens:\s*(\d+)", config)}, {engine})
        models = json.loads(read("app/pi/agent/models.json"))
        self.assertEqual({m["contextWindow"] for p in models["providers"].values() for m in p["models"]}, {engine})

    def test_a_full_prompt_plus_a_full_answer_fits_the_context(self):
        engine = int(yaml.safe_load(read("cluster/sglang/config.yaml"))["data"]["MAX_MODEL_LEN"])
        self.assertLess(guard.MAX_OUTPUT_TOKENS, engine // 4)      # the answer never takes more than a quarter of the window


if __name__ == "__main__":
    unittest.main()
