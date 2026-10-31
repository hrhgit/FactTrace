import asyncio
from argparse import Namespace
import json
import unittest

import httpx

from text_memory.answer_templates import configuration, load_pipeline, official_messages
from text_memory.model import HttpJsonModel, ModelConfig


class OfficialAnswerTests(unittest.TestCase):
    def test_official_messages_render_the_pinned_template(self):
        context = 'first\n{"text":"second {{question}}"}'
        for dataset in ("longmemeval", "locomo-refined"):
            expected = load_pipeline(dataset).render_answer_prompt({"question": "Which?", "retrieved_context": context})
            self.assertEqual(official_messages("Which?", context, dataset), [{"role": "user", "content": expected}])
            self.assertEqual(configuration("official", dataset)["answer_parameters"], {"temperature": 0})
        with self.assertRaises(Exception):
            configuration("project", "longmemeval")


    def test_http_matches_upstream_payload_without_length_or_response_format(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "test", "choices": [
                {"finish_reason": "stop", "message": {"content": "answer"}}]})
        model = HttpJsonModel(ModelConfig("https://test.invalid", "test", "test"), transport=httpx.MockTransport(handler))
        messages = [{"role": "user", "content": "question"}]
        model.complete_text(messages, purpose="answer", **configuration("official", "longmemeval")["answer_parameters"])
        self.assertEqual(bodies[0], {"model": "test", "messages": messages, "temperature": 0})
        async def upstream(dataset):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await load_pipeline(dataset).complete(client, Namespace(
                    base_url="https://test.invalid", api_key="test", model="test"), "question", 256)
        for dataset in ("longmemeval", "locomo-refined"):
            self.assertEqual(asyncio.run(upstream(dataset)), "answer")
            self.assertEqual(bodies[-1], bodies[0])
        model.complete_text(messages, purpose="answer")
        self.assertEqual(bodies[-1], {"model": "test", "messages": messages})

    def test_official_grading_contract_for_both_datasets(self):
        for dataset in ("longmemeval", "locomo-refined"):
            pipeline = load_pipeline(dataset)
            self.assertIn("Gold answer: A\nB", pipeline.render_accuracy_prompt(
                {"question": "Which?", "gold_answer": ["A", "B"]}, "A and B"))
            self.assertEqual(pipeline.parse_judge_label('{"label":"CORRECT"}'), "CORRECT")
            self.assertEqual(pipeline.parse_judge_label('{"label":"WRONG"}'), "WRONG")
            with self.assertRaises(ValueError):
                pipeline.parse_judge_label('{"label":"unknown"}')

    def test_locomo_refined_judge_prompt_is_concrete_json_and_parser_handles_fence_noise(self):
        pipeline = load_pipeline("locomo-refined")
        prompt = pipeline.render_accuracy_prompt(
            {"question": "Which?", "gold_answer": "A"}, "A"
        )
        self.assertIn('{"label": "CORRECT"}', prompt)
        self.assertNotIn('{{', prompt)
        self.assertNotIn('"CORRECT" or "WRONG"', prompt)
        self.assertEqual(
            pipeline.parse_judge_label(
                'Reason: it matches.\n```json\n{{"label": "CORRECT"}}\n```'
            ),
            "CORRECT",
        )
