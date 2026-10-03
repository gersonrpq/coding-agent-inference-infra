"""Guard: what dies before it can reach admission or a GPU."""
import unittest
from unittest import mock

from fastapi import HTTPException

from tests._env import guard

HOOK = guard.security_inspect
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def request(**extra):
    data = {"model": "qwen-coding-local", "messages": [{"role": "user", "content": "hi"}]}
    data.update(extra)
    return data


async def run(data, call_type="acompletion"):
    return await HOOK.async_pre_call_hook(None, None, data, call_type)


class Rejections(unittest.IsolatedAsyncioTestCase):
    async def assertRejected(self, data, status, code):
        with self.assertRaises(HTTPException) as caught:
            await run(data)
        exc = caught.exception
        self.assertEqual(exc.status_code, status)
        self.assertEqual(exc.detail["error"]["code"], code)
        self.assertEqual(exc.headers["source"], "inspect")

    async def test_valid_request_passes(self):
        data = await run(request(max_tokens=512))
        self.assertEqual(data["model"], "qwen-coding-local")

    async def test_unknown_model_is_403(self):
        await self.assertRejected(request(model="gpt-4o"), 403, "model_not_allowed")

    async def test_unknown_role_is_400(self):
        await self.assertRejected(request(messages=[{"role": "wizard", "content": "x"}]), 400, "invalid_role")

    async def test_every_allowed_role_passes(self):
        for role in ("system", "developer", "user", "assistant", "tool"):
            await run(request(messages=[{"role": role, "content": "x"}]))

    async def test_text_parts_given_as_a_list_pass(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        await run(request(messages=[{"role": "user", "content": content}]))

    async def test_a_data_uri_image_passes(self):
        content = [{"type": "text", "text": "what colour?"}, {"type": "image_url", "image_url": {"url": PNG}}]
        await run(request(messages=[{"role": "user", "content": content}]))
        await run(request(messages=[{"role": "tool", "content": [{"type": "image_url", "image_url": PNG}]}]))   # url as a plain string too

    async def test_remote_image_urls_are_refused(self):
        content = [{"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data"}}]
        await self.assertRejected(request(messages=[{"role": "user", "content": content}]), 400, "remote_image_not_allowed")

    async def test_audio_video_and_files_are_not_supported(self):
        for kind in ("input_audio", "video_url", "file"):
            await self.assertRejected(request(messages=[{"role": "user", "content": [{"type": kind}]}]), 400, "multimodal_not_allowed")

    async def test_images_only_in_user_and_tool_messages(self):
        for role in ("system", "assistant", "developer"):
            content = [{"type": "image_url", "image_url": {"url": PNG}}]
            await self.assertRejected(request(messages=[{"role": role, "content": content}]), 400, "invalid_content_part")

    async def test_image_type_count_and_size_limits(self):
        gif = PNG.replace("image/png", "image/tiff")
        await self.assertRejected(request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": gif}}]}]), 400, "unsupported_image_type")
        many = [{"type": "image_url", "image_url": {"url": PNG}}] * 9
        await self.assertRejected(request(messages=[{"role": "user", "content": many}]), 413, "too_many_images")
        big = "data:image/png;base64," + "A" * (7 * 1024 * 1024)               # ~5.25 MiB decoded
        await self.assertRejected(request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": big}}]}]), 413, "image_too_large")
        total = "data:image/png;base64," + "A" * (4 * 1024 * 1024)             # 3 MiB each, 6 of them = 18 MiB
        await self.assertRejected(request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": total}}] * 6}]), 413, "images_too_large")

    async def test_images_can_be_switched_off_but_text_parts_still_pass(self):
        with mock.patch.object(guard, "ALLOW_MULTIMODAL", False):
            await self.assertRejected(request(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG}}]}]), 400, "multimodal_not_allowed")
            await run(request(messages=[{"role": "user", "content": [{"type": "text", "text": "x"}]}]))

    async def test_tools_must_be_a_list_of_objects(self):
        await self.assertRejected(request(tools="nope"), 400, "invalid_tools")
        await self.assertRejected(request(tools=["nope"]), 400, "invalid_tool")

    async def test_too_many_tools_is_413(self):
        await self.assertRejected(request(tools=[{"type": "function"}] * 65), 413, "too_many_tools")
        await run(request(tools=[{"type": "function"}] * 64))

    async def test_oversized_tool_schema_is_413(self):
        big = {"type": "function", "function": {"description": "x" * (70 * 1024)}}
        await self.assertRejected(request(tools=[big]), 413, "tool_schema_too_large")

    async def test_max_tokens_rules(self):
        with mock.patch.object(guard, "MAX_TOKENS_POLICY", "reject"):
            await self.assertRejected(request(max_tokens=4097), 400, "max_tokens_exceeded")
        await self.assertRejected(request(max_tokens=0), 400, "invalid_max_tokens")
        await self.assertRejected(request(max_tokens=-5), 400, "invalid_max_tokens")
        await self.assertRejected(request(max_tokens="100"), 400, "invalid_max_tokens")
        await self.assertRejected(request(max_tokens=True), 400, "invalid_max_tokens")
        await run(request(max_tokens=4096))
        await run(request())                                   # no limit given: the model default applies

    async def test_the_default_policy_is_clamp(self):
        self.assertEqual(guard.MAX_TOKENS_POLICY, "clamp")
        self.assertEqual((await run(request(max_tokens=16384)))["max_tokens"], 4096)

    async def test_clamp_policy_serves_a_big_request_with_the_cap(self):
        with mock.patch.object(guard, "MAX_TOKENS_POLICY", "clamp"):
            data = await run(request(max_tokens=16384))
            self.assertEqual(data["max_tokens"], 4096)
            data = await run(request(max_completion_tokens=20000, max_tokens=100))
            self.assertEqual((data["max_completion_tokens"], data["max_tokens"]), (4096, 100))

    async def test_clamp_policy_still_refuses_nonsense(self):
        with mock.patch.object(guard, "MAX_TOKENS_POLICY", "clamp"):
            await self.assertRejected(request(max_tokens=0), 400, "invalid_max_tokens")
            await self.assertRejected(request(max_tokens="9000"), 400, "invalid_max_tokens")

    async def test_max_completion_tokens_takes_precedence(self):
        with mock.patch.object(guard, "MAX_TOKENS_POLICY", "reject"):
            await self.assertRejected(request(max_completion_tokens=5000, max_tokens=10), 400, "max_tokens_exceeded")


class Behaviour(unittest.IsolatedAsyncioTestCase):
    async def test_default_priority_is_interactive_and_a_batch_priority_is_kept(self):
        self.assertEqual((await run(request()))["priority"], 5)
        self.assertEqual((await run(request(priority=8)))["priority"], 8)

    async def test_interactive_priorities_are_normalised_so_nobody_jumps_the_queue(self):
        for value in (1, 3, 5):
            self.assertEqual((await run(request(priority=value)))["priority"], 5)

    async def assertCode(self, data, code):
        with self.assertRaises(HTTPException) as caught:
            await run(data)
        self.assertEqual((caught.exception.status_code, caught.exception.detail["error"]["code"]), (400, code))

    async def test_priority_out_of_range_or_not_an_integer_is_400(self):
        for value in (0, -1000, 11, 5.5, "5", True):
            await self.assertCode(request(priority=value), "invalid_priority")

    async def test_several_choices_are_refused_because_they_cost_several_sequences(self):
        for key in ("n", "best_of"):
            await self.assertCode(request(**{key: 4}), "multiple_choices_not_supported")
        await run(request(n=1))

    async def test_other_endpoints_are_not_inspected(self):
        data = request(model="not-allowed")
        self.assertIs(await run(data, call_type="embedding"), data)

    async def test_thinking_switch_is_not_blocked(self):
        data = request(chat_template_kwargs={"enable_thinking": False})
        self.assertEqual((await run(data))["chat_template_kwargs"], {"enable_thinking": False})

    async def test_unexpected_error_fails_closed_with_503(self):
        with mock.patch.object(guard, "inspect", side_effect=RuntimeError("bug")):
            with self.assertRaises(HTTPException) as caught:
                await run(request())
        exc = caught.exception
        self.assertEqual(exc.status_code, 503)
        self.assertEqual(exc.detail["error"]["code"], "security_inspection_failed")
        self.assertEqual(exc.headers["source"], "inspect")

    async def test_caller_id_never_exposes_the_key(self):
        class Caller:
            user_id = "alice-secret-id"
        ident = guard._caller_id(Caller())
        self.assertNotIn("alice", ident)
        self.assertEqual(len(ident), 12)
        self.assertEqual(guard._caller_id(None), "anonymous")


if __name__ == "__main__":
    unittest.main()
