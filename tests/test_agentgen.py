"""The agentic generator: its tools are confined to the repository, it assembles streamed tool calls, and a session really
loops model -> tool -> model until the model answers."""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "script"))
spec = importlib.util.spec_from_file_location("agentgen", os.path.join(HERE, "script", "agentgen.py"))
agentgen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentgen)


class Tools(unittest.TestCase):
    def test_read_file_pages_and_numbers_lines(self):
        out = agentgen.run_tool("read_file", {"path": "control/place.py", "offset": 1, "limit": 3})
        self.assertTrue(out.splitlines()[0].lstrip().startswith("1  "))
        self.assertEqual(len(out.splitlines()), 3)

    def test_a_long_result_is_truncated_with_a_hint(self):
        out = agentgen.run_tool("read_file", {"path": "ARCHITECTURE.md", "limit": 100000})
        self.assertLessEqual(len(out), agentgen.READ_CHARS + 200)
        self.assertIn("truncated", out)

    def test_paths_outside_the_repository_and_data_folders_are_refused(self):
        for path in ("../x", "/etc/passwd", "metrics/runs", ".git/config"):
            self.assertTrue(agentgen.run_tool("read_file", {"path": path}).startswith("error:"), path)
        self.assertTrue(agentgen.run_tool("list_dir", {"path": ".."}).startswith("error:"))

    def test_list_dir_marks_directories_and_hides_data_folders(self):
        out = agentgen.run_tool("list_dir", {"path": "."}).splitlines()
        self.assertIn("control/", out)
        self.assertNotIn("metrics/", out)
        self.assertNotIn(".git/", out)

    def test_grep_finds_code_and_reports_no_matches_and_bad_patterns(self):
        self.assertIn("control/place.py", agentgen.run_tool("grep", {"pattern": "def pick", "path": "control"}))
        self.assertEqual(agentgen.run_tool("grep", {"pattern": "zzzz_not_there_zzzz", "path": "control"}), "no matches")
        self.assertTrue(agentgen.run_tool("grep", {"pattern": "(", "path": "control"}).startswith("error:"))

    def test_unknown_tools_and_missing_arguments_come_back_as_text(self):
        self.assertIn("unknown tool", agentgen.run_tool("rm", {}))
        self.assertTrue(agentgen.run_tool("read_file", {}).startswith("error:"))


def sse(*events):
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


class FakeGateway(BaseHTTPRequestHandler):
    """First call of a turn asks for a tool; once a tool result is in the history, it answers."""
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGateway.requests.append(body)
        usage = {"usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 800}}, "choices": []}
        if body["messages"][-1]["role"] == "tool":
            events = [{"choices": [{"delta": {"content": "The answer "}}]}, {"choices": [{"delta": {"content": "is 42."}, "finish_reason": "stop"}]}, usage]
        else:
            events = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\": \"READ"}}]}}]},
                      {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ME.md\"}"}}]}, "finish_reason": "tool_calls"}]}, usage]
        data = sse(*events)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-litellm-model-api-base", "http://sglang-worker-0:30000/v1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Loop(unittest.TestCase):
    def setUp(self):
        FakeGateway.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), FakeGateway)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def test_streamed_tool_call_fragments_are_assembled(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        rec = agentgen.agent_call("127.0.0.1", self.port, "k", body, 10)
        self.assertEqual(rec["status"], 200)
        self.assertEqual(rec["tool_calls"], [{"id": "call_1", "name": "read_file", "arguments": "{\"path\": \"README.md\"}"}])
        self.assertIsNotNone(rec["ttft"])
        self.assertEqual((rec["prompt_tokens"], rec["cached_tokens"], rec["finish"]), (1000, 800, "tool_calls"))
        self.assertTrue(rec["worker"].endswith("30000/v1"))

    def test_a_session_loops_model_tool_model_until_the_answer(self):
        os.environ["LITELLM_API_KEY"] = "test-key"
        self.addCleanup(lambda: os.environ.pop("LITELLM_API_KEY", None))
        out = tempfile.mkdtemp()
        a = argparse.Namespace(url=f"http://127.0.0.1:{self.port}", model="qwen-coding-local", sessions=1, duration=2.0, warmup=0.0, ramp=0.0,
                               batch_share=0.0, max_steps=6, max_tokens=4096, max_prompt=60000, tool_time=0.1, turn_idle=0.5, retries=0,
                               timeout=10, slo_ttft=20.0, max_shed=0.01, seed=1, out=out, profile="agent", tok_per_word=3.9, system_tokens=100)
        summary = agentgen.AgentRun(a).go()
        agent = summary["agent"]
        self.assertGreaterEqual(agent["turns_finished"], 1)
        self.assertEqual(agent["steps_per_finished_turn_p50"], 2)               # one tool call, then the answer
        self.assertEqual(agent["tool_calls_by_name"]["read_file"], agent["turns_finished"])
        second = [r for r in FakeGateway.requests if r["messages"][-1]["role"] == "tool"][0]
        self.assertEqual(second["messages"][-2]["tool_calls"][0]["function"]["name"], "read_file")   # the history carries the call
        self.assertIn("tools", second)                                                  # real tool schemas are sent
        self.assertIn("CLAUDE.md", second["messages"][0]["content"])                    # the shared prefix is a real file
        rows = [json.loads(l) for l in open(os.path.join(out, "records.jsonl"))]
        self.assertTrue(all(r["status"] == 200 for r in rows))
        self.assertIn(["read_file"], [r["tools"] for r in rows])
        for name in ("summary.json", "timeline.csv", "config.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)))


if __name__ == "__main__":
    unittest.main()
