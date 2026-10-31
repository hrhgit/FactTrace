import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from scripts import run_mixed_answers
from text_memory.errors import ModelError
from text_memory.evaluation import TimedModel
from text_memory.mixed_eval import answer_messages, grade, pipeline
from text_memory.model import ModelConfig
from tests.fixtures import ScriptedModel

TASKS = {
    "locomo-refined": {"dataset": "locomo-refined", "category": "temporal"},
    "longmemeval": {"dataset": "longmemeval", "category": "multi-session"},
    "beam": {"dataset": "beam", "category": "information_extraction"},
    "clbench": {"dataset": "clbench", "category": "Rule System Application", "system_prompt": "SYSTEM RULES"},
    "personamem-v1": {"dataset": "personamem-v1", "category": "recall_user_shared_facts",
                      "all_options": "['(a) tea', '(b) coffee', '(c) juice', '(d) water']"},
    "personamem-v2": {"dataset": "personamem-v2", "category": "stereotypical_pref",
                      "options": ["Option zero", "Option one", "Option two", "Option three"]},
}


class AnswerMessageTests(unittest.TestCase):
    def test_every_benchmark_uses_its_official_renderer_with_the_context(self):
        for dataset in ("locomo-refined", "longmemeval"):
            expected = pipeline(dataset).render_answer_prompt({"question": "When?", "retrieved_context": "MEMORY"})
            self.assertEqual(answer_messages(TASKS[dataset], "When?", "MEMORY", "m"), [{"role": "user", "content": expected}])
        beam = answer_messages(TASKS["beam"], "What?", "MEMORY", "m")[0]["content"]
        self.assertEqual(beam, pipeline("beam").render_answer_prompt({"question": "What?", "context": "MEMORY"}))
        clbench = answer_messages(TASKS["clbench"], "Do it.", "MEMORY", "m")[0]["content"]
        # The pinned template itself begins with a literal backslash line.
        self.assertTrue(clbench.startswith("\\\nSYSTEM RULES") and "MEMORY" in clbench and clbench.endswith("Do it."))
        self.assertIn("(no memories)", answer_messages(TASKS["clbench"], "Do it.", "", "m")[0]["content"])

    def test_personamem_context_is_one_user_message_and_v2_options_keep_the_set_order(self):
        v1 = answer_messages(TASKS["personamem-v1"], "What do I drink?", "MEMORY", "m")
        self.assertEqual(v1[0], {"role": "user", "content": "MEMORY"})
        self.assertTrue(v1[-1]["content"].endswith(TASKS["personamem-v1"]["all_options"]))
        self.assertEqual(len(answer_messages(TASKS["personamem-v1"], "What do I drink?", "", "m")), 1)
        v2 = answer_messages(TASKS["personamem-v2"], "Suggest?", "MEMORY", "m")
        self.assertEqual(v2[0], {"role": "user", "content": "MEMORY"})
        self.assertTrue(v2[1]["content"].startswith("Suggest?") and v2[1]["content"].endswith(pipeline("personamem-v2").RECALL_SUFFIX))
        self.assertEqual(v2[2]["role"], "system")
        self.assertIn("A. Option zero\nB. Option one\nC. Option two\nD. Option three", v2[2]["content"])


class GradeTests(unittest.TestCase):
    def test_open_ended_benchmarks_use_the_official_judge(self):
        judge = ScriptedModel('{"label": "CORRECT"}', '{"label": "WRONG"}')
        self.assertEqual(grade(TASKS["locomo-refined"], "How many?", {"gold_answer": ["two"]}, "2", judge)["score"], 1.0)
        self.assertEqual(grade(TASKS["longmemeval"], "Where?", {"gold_answer": "Paris"}, "Rome", judge)["score"], 0.0)
        self.assertIn("two", judge.calls[0]["messages"][0]["content"])

    def test_beam_scores_the_mean_rubric_and_adds_ordering_metrics(self):
        scores = {"scores": [{"index": 0, "score": 1.0, "reason": "ok"}, {"index": 1, "score": 0.5, "reason": "partly"}]}
        result = grade(TASKS["beam"], "Q", {"question_type": "information_extraction", "rubric": ["a", "b"]}, "A",
                       ScriptedModel(json.dumps(scores)))
        self.assertEqual(result["score"], 0.75)
        ordering = {"question_type": "event_ordering", "rubric": ["first", "second"]}
        # Candidate "2nd" matches rubric 1 on the second try; "1st" then matches rubric 0.
        judge = ScriptedModel(json.dumps(scores), "NO", "YES", "YES")
        result = grade(TASKS["beam"] | {"category": "event_ordering"}, "Q", ordering, "2nd\n1st", judge)
        self.assertEqual(result["event_ordering"], pipeline("beam").event_ordering_metrics(["first", "second"], ["second", "first"]))
        self.assertEqual(judge.calls[1]["messages"][0]["content"], pipeline("beam").EQUIVALENCE_SYSTEM_PROMPT)

    def test_invalid_judge_output_raises_instead_of_scoring_zero(self):
        for dataset, label in (("beam", {"question_type": "x", "rubric": ["a"]}),
                               ("clbench", {"rubrics": ["r"]})):
            with self.subTest(dataset=dataset):
                judge = ScriptedModel("not json 1", "not json 2", "not json 3")
                with self.assertRaisesRegex(ModelError, "judge.*3 次输出尝试"):
                    grade(TASKS[dataset], "Q", label, "A", judge)
                self.assertEqual(len(judge.calls), 3)
                self.assertFalse(judge.calls[-1]["output_validation"]["retry_scheduled"])

    def test_bad_judgments_receive_feedback_and_can_recover(self):
        score = '{"scores":[{"index":0,"score":1,"reason":"correct"}]}'
        clbench = '{"Overall Score":1,"List of Requirement Satisfaction Status":["yes"]}'
        cases = [
            ("locomo-refined", {"gold_answer": "a"}, '{"label":"maybe"}', '{"label":"CORRECT"}'),
            ("longmemeval", {"gold_answer": "a"}, "not json", '{"label":"CORRECT"}'),
            ("beam", {"question_type": "x", "rubric": ["a"]}, '{"scores":[]}', score),
            ("clbench", {"rubrics": ["r"]}, '{"Overall Score":"unknown"}', clbench),
            ("clbench", {"rubrics": ["r"]}, '{"Overall Score":0}', clbench),
            ("clbench", {"rubrics": ["r"]}, '{"Overall Score":1,"List of Requirement Satisfaction Status":["maybe"]}', clbench),
        ]
        for dataset, label, bad, good in cases:
            with self.subTest(dataset=dataset, bad=bad):
                judge = ScriptedModel(bad, good)
                self.assertEqual(grade(TASKS[dataset], "Q", label, "A", judge)["score"], 1.0)
                self.assertEqual(len(judge.calls), 2)
                first, second = [call["messages"] for call in judge.calls]
                self.assertEqual(second[:-2], first)
                self.assertEqual(second[-2], {"role": "assistant", "content": bad})
                self.assertIn("tool_error", json.loads(second[-1]["content"]))
                self.assertTrue(judge.calls[0]["output_validation"]["retry_scheduled"])

    def test_event_alignment_does_not_silently_treat_bad_output_as_no(self):
        scores = '{"scores":[{"index":0,"score":1,"reason":"ok"}]}'
        label = {"question_type": "event_ordering", "rubric": ["event"]}
        judge = ScriptedModel(scores, "unclear", "YES")
        result = grade(TASKS["beam"], "Q", label, "answer", judge)
        self.assertEqual(result["event_ordering"], pipeline("beam").event_ordering_metrics(["event"], ["event"]))
        self.assertEqual(len(judge.calls), 3)
        self.assertIn("YES 或 NO", judge.calls[1]["output_validation"]["message"])
        with self.assertRaises(ModelError):
            grade(TASKS["beam"], "Q", label, "answer", ScriptedModel(scores, "unclear", "unclear"))

    def test_judge_program_errors_are_not_retried(self):
        judge = ScriptedModel('{"scores":[]}')
        with patch.object(pipeline("beam"), "parse_rubric_scores", side_effect=TypeError("local defect")):
            with self.assertRaisesRegex(TypeError, "local defect"):
                grade(TASKS["beam"], "Q", {"question_type": "x", "rubric": ["a"]}, "A", judge)
        self.assertEqual(len(judge.calls), 1)

    def test_clbench_is_all_or_nothing_and_empty_answers_skip_the_judge(self):
        payload = {"Grading Rationale": "missed one", "List of Requirement Satisfaction Status": ["yes", "no"], "Overall Score": 0}
        result = grade(TASKS["clbench"], "Q", {"rubrics": ["r1", "r2"]}, "Final Answer: text",
                       ScriptedModel("```json\n" + json.dumps(payload) + "\n```"))
        self.assertEqual((result["score"], result["requirement_ratio"]), (0.0, 0.5))
        judge = ScriptedModel()
        self.assertEqual(grade(TASKS["clbench"], "Q", {"rubrics": ["r"]}, "  ", judge)["score"], 0.0)
        self.assertEqual(judge.calls, [])

    def test_multiple_choice_needs_no_judge(self):
        self.assertEqual(grade(TASKS["personamem-v1"], "Q", {"correct_answer": "(b)"}, "<final_answer> (b)", None)["score"], 1.0)
        self.assertEqual(grade(TASKS["personamem-v1"], "Q", {"correct_answer": "(b)"}, "(a) or (b)", None)["score"], 0.0)
        self.assertEqual(grade(TASKS["personamem-v2"], "Q", {"correct_letter": "C"}, "Reasoning.\nFinal Answer: C", None)["score"], 1.0)


class RunnerTests(unittest.TestCase):
    def write_set(self, root: Path, ids):
        (root / "inputs").mkdir(parents=True)
        (root / "scoring").mkdir()
        rows = lambda values: "".join(json.dumps(v) + "\n" for v in values)
        (root / "inputs/queries.jsonl").write_text(rows({"id": i, "user_id": "u", "question": "Q " + i} for i in ids))
        (root / "inputs/tasks.jsonl").write_text(rows({"id": i, **TASKS["personamem-v2"]} for i in ids))
        (root / "scoring/labels.jsonl").write_text(rows({"id": i, "correct_letter": "B"} for i in ids))

    def test_answer_grade_and_subset_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "big", ["x:1", "x:2", "x:3"])
            self.write_set(root / "small", ["x:1"])
            (root / "contexts.jsonl").write_text("".join(json.dumps({"id": i, "context": "C"}) + "\n" for i in ["x:1", "x:2", "x:3"]))
            config = ModelConfig(base_url="http://local", api_key="k", model="m")
            answers = iter(["Final Answer: B", "Final Answer: A", "Final Answer: B"])
            out = root / "run"
            with patch.object(run_mixed_answers.ModelConfig, "from_env", return_value=config), \
                 patch.object(run_mixed_answers, "TimedModel",
                              side_effect=lambda c, channel_pool=None: ScriptedModel(next(answers, "unused"))):
                main = lambda *argv: (patch("sys.argv", ["run", *argv]).start(), run_mixed_answers.main())[1]
                # Labels are not needed to answer.
                (root / "big/scoring/labels.jsonl").rename(root / "labels.bak")
                self.assertEqual(main("answer", "--set", str(root / "big"), "--contexts", str(root / "contexts.jsonl"),
                                      "--output", str(out), "--workers", "1"), 0)
                (root / "labels.bak").rename(root / "big/scoring/labels.jsonl")
                self.assertEqual(main("grade", "--output", str(out)), 0)
                self.assertEqual(main("summary", "--output", str(out), "--subset", str(root / "small")), 0)
                patch.stopall()
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["all"]["datasets"]["personamem-v2"]["sum"], 2.0)
            self.assertEqual((summary["subset"]["inside"]["questions"], summary["subset"]["inside"]["micro_score"]), (1, 1.0))
            self.assertEqual(summary["subset"]["outside"]["micro_score"], 0.5)
            self.assertIn("labels_sha256", json.loads((out / "manifest.json").read_text())["judge"])

    def test_failed_answers_and_grades_keep_traces_without_creating_results_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", ["x:1"])
            (root / "set/inputs/tasks.jsonl").write_text(json.dumps({"id": "x:1", **TASKS["beam"]}) + "\n")
            (root / "set/scoring/labels.jsonl").write_text(json.dumps(
                {"id": "x:1", "question_type": "information_extraction", "rubric": ["answer"]}) + "\n")
            (root / "contexts.jsonl").write_text('{"id":"x:1","context":"C"}\n')
            out = root / "run"
            config = ModelConfig("https://test.invalid", "test-key", "test-model")
            state = {"response": "  ", "count": 0}

            def handler(request):
                state["count"] += 1
                content = (f"bad-{state['count']}" if state["response"] == "bad" else state["response"])
                return httpx.Response(200, json={"usage": {"total_tokens": 3}, "choices": [
                    {"finish_reason": "stop", "message": {"content": content}}]})

            def main(*argv):
                with patch("sys.argv", ["run", *argv]):
                    return run_mixed_answers.main()

            transport = httpx.MockTransport(handler)
            with patch.object(run_mixed_answers.ModelConfig, "from_env", return_value=config), \
                 patch.object(run_mixed_answers, "TimedModel", side_effect=lambda c, channel_pool=None: TimedModel(
                     c, channel_pool=channel_pool, transport=transport, sleep=lambda _: None)), \
                 patch.object(run_mixed_answers, "print", create=True):
                args = ("answer", "--set", str(root / "set"), "--contexts", str(root / "contexts.jsonl"),
                        "--output", str(out), "--workers", "1")
                self.assertEqual(main(*args), 1)
                self.assertFalse((out / "answers/x_1.json").exists())
                trace = json.loads((out / "traces/answer-x_1.json").read_text())
                self.assertEqual(len(trace["calls"]), 3)
                self.assertTrue(all(c["status"] == "invalid_output" for c in trace["calls"]))
                state["response"] = "answer"
                self.assertEqual(main(*args), 0)
                self.assertEqual(json.loads((out / "answers/x_1.json").read_text())["generated_answer"], "answer")
                state["response"] = "bad"
                self.assertEqual(main("grade", "--output", str(out)), 1)
                self.assertFalse((out / "grades/x_1.json").exists())
                trace = json.loads((out / "traces/grade-x_1.json").read_text())
                self.assertEqual(len(trace["calls"]), 3)
                self.assertTrue(all("output_validation" in c for c in trace["calls"]))
                state["response"] = '{"scores":[{"index":0,"score":1,"reason":"ok"}]}'
                self.assertEqual(main("grade", "--output", str(out)), 0)
                self.assertEqual(json.loads((out / "grades/x_1.json").read_text())["score"], 1.0)
                errors = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
                self.assertEqual([error["phase"] for error in errors], ["answer", "grade"])


if __name__ == "__main__":
    unittest.main()
