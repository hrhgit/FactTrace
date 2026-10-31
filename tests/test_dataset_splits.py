from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from text_memory.datasets import prepare_locomo, prepare_longmemeval, sample_longmemeval, split_locomo
from text_memory.errors import ValidationError
from text_memory.evaluation import jsonl


class DatasetSplitTests(unittest.TestCase):
    def test_stratified_deterministic_sample_keeps_original_and_excludes_paired_versions(self):
        rows = [{"question_id": f"{kind}-{i}" + suffix, "question_type": kind}
                for kind in ("a", "b", "c", "d", "e", "f") for i in range(20) for suffix in ("", "_abs")]
        mandatory = {"a-0", "a-1_abs", "b-0"}
        kwargs = dict(seed=20261023, dev_per_type=8, holdout_per_type=3, unanswerable=5)
        report = sample_longmemeval(rows, mandatory, **kwargs)
        self.assertEqual(report, sample_longmemeval(list(reversed(rows)), mandatory, **kwargs))
        self.assertEqual(set(report["dev_counts"].values()), {8})
        self.assertEqual(set(report["holdout_counts"].values()), {3})
        self.assertEqual(report["dev_unanswerable"], 5)
        self.assertTrue(mandatory <= set(report["dev"]))
        self.assertFalse({i.removesuffix("_abs") for i in report["dev"]} &
                         {i.removesuffix("_abs") for i in report["holdout"]})

    def locomo(self):
        return {"sample_id": "conv-test", "event_summary": "SECRET_SUMMARY", "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob", "session_2_date_time": "2026-03-02",
            "session_2": [{"dia_id": "D2:1", "speaker": "Bob", "text": "I finished it."}],
            "session_1_date_time": "2026-03-01", "session_1": [
                {"dia_id": "D1:1", "speaker": "Alice", "text": "The panel.", "blip_caption": "A red panel."}]},
            "qa": [{"question": "Who?", "answer": "SECRET_GOLD", "evidence": ["D2:1"], "category": 1},
                   {"question": "What?", "answer": ["red", "panel"], "evidence": ["D1:1"], "category": 4}]}

    def test_locomo_preparation_preserves_dates_speakers_all_questions_and_labels(self):
        original = self.locomo()
        before = deepcopy(original)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            report = prepare_locomo([original], output, ids={"conv-test"}, media_text="captions")
            histories = jsonl(output / "histories.jsonl")
            self.assertEqual([h["session_id"] for h in histories], ["session_1", "session_2"])
            self.assertEqual(histories[1]["messages"][0]["speaker"], "Bob")
            self.assertEqual(histories[1]["messages"][0]["session_date"], "2026-03-02")
            self.assertIn("A red panel.", histories[0]["messages"][0]["content"])
            inputs = (output / "histories.jsonl").read_text() + (output / "queries.jsonl").read_text()
            for forbidden in ("SECRET_GOLD", "SECRET_SUMMARY", '"evidence"', '"answer"', "D2:1"):
                self.assertNotIn(forbidden, inputs)
            labels = jsonl(output / "labels.jsonl")
            self.assertEqual(labels[0]["answer_turns"], [{"session_id": "session_2", "position": 0}])
            self.assertEqual(labels[1]["accepted_answers"], ["red\npanel"])
            self.assertEqual(report["cases"], 2)
        self.assertEqual(original, before)

    def test_holdout_needs_no_answer_or_evidence_fields(self):
        row = self.locomo()
        for q in row["qa"]:
            del q["answer"], q["evidence"]
        lme = {"question_id": "q", "question": "Who?", "question_type": "one",
               "haystack_session_ids": ["s"], "haystack_dates": ["date"],
               "haystack_sessions": [[{"role": "user", "content": "fact", "has_answer": True}]]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            prepare_locomo([row], output / "loco", ids={"conv-test"}, media_text="captions", include_labels=False)
            prepare_longmemeval([lme], output / "lme", include_labels=False)
            self.assertFalse(list(output.glob("*/labels.jsonl")))
            self.assertNotIn("has_answer", (output / "lme/histories.jsonl").read_text())

    def test_invalid_evidence_is_reported_without_partial_files(self):
        row = self.locomo()
        row["qa"][0]["evidence"] = ["D99:1"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad"
            with self.assertRaises(ValidationError):
                prepare_locomo([row], output, ids={"conv-test"}, media_text="none")
            self.assertFalse(output.exists())

    def test_semicolon_separated_evidence_is_resolved_and_raw_annotation_kept(self):
        row = self.locomo()
        row["qa"][0]["evidence"] = ["D1:1; D2:1"]
        with tempfile.TemporaryDirectory() as directory:
            report = prepare_locomo([row], Path(directory), ids={"conv-test"}, media_text="none")
            label = jsonl(Path(directory) / "labels.jsonl")[0]
            self.assertEqual(label["evidence"], ["D1:1; D2:1"])
            self.assertEqual(label["resolved_evidence"], ["D1:1", "D2:1"])
            self.assertEqual(len(label["answer_turns"]), 2)
            self.assertEqual(report["semicolon_evidence_questions"], ["conv-test#q0000"])

    def test_conversation_split_is_deterministic_and_disjoint(self):
        rows = [{"sample_id": str(i)} for i in range(10)]
        report = split_locomo(rows, seed=20261023, dev_conversations=3)
        self.assertEqual(report, split_locomo(rows[::-1], seed=20261023, dev_conversations=3))
        self.assertEqual((len(report["dev"]), len(report["holdout"])), (3, 7))
        self.assertFalse(set(report["dev"]) & set(report["holdout"]))
