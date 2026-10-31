"""Program-only time ranges for search: periods a question names and the times of a fact."""
from datetime import date
import unittest

from text_memory.collections.time_ranges import fact_ranges, overlaps, question_ranges


def d(value):
    return date.fromisoformat(value)


class QuestionRangeTests(unittest.TestCase):
    def check(self, question, *expected):
        self.assertEqual(question_ranges(question), [(d(a), d(b)) for a, b in expected])

    def test_days_months_and_years(self):
        self.check("What did Gina find on 1 February, 2023?", ("2023-02-01", "2023-02-01"))
        self.check("What did Gina get accepted for on May 23, 2023?", ("2023-05-23", "2023-05-23"))
        self.check("What class did Maria start in December 2023?", ("2023-12-01", "2023-12-31"))
        self.check("Where did John travel in 2022?", ("2022-01-01", "2022-12-31"))

    def test_modifiers(self):
        self.check("What did Maria do last weekend before April 10, 2023?", ("2023-04-08", "2023-04-09"))
        self.check("What did John do the week before August 3, 2023?", ("2023-07-27", "2023-08-02"))
        self.check("Which puppy did she get two weeks before August 11, 2023?", ("2023-07-25", "2023-07-31"))
        self.check("What did they do on the Monday before July 24, 2023?", ("2023-07-17", "2023-07-17"))
        self.check("Where was Calvin in the last week of October 2023?", ("2023-10-25", "2023-10-31"))
        self.check("What did Dave attend on the first Friday of October 2023?", ("2023-10-06", "2023-10-06"))
        self.check("Where was John between August 11 and August 15 2023?", ("2023-08-11", "2023-08-15"))

    def test_no_period(self):
        self.assertEqual(question_ranges("Which outdoor spot did Joanna visit in May?"), [])
        self.assertEqual(question_ranges("When did Jon visit networking events?"), [])


class FactRangeTests(unittest.TestCase):
    def test_session_day_and_resolved_relative_dates(self):
        spans = fact_ranges("Dave opened his car shop last week.", ["1:56 pm on 16 May, 2023"])
        self.assertEqual(spans, [(d("2023-05-16"), d("2023-05-16")), (d("2023-05-08"), d("2023-05-14"))])

    def test_written_dates_without_session(self):
        self.assertEqual(fact_ranges("He moved in March 2022.", []), [(d("2022-03-01"), d("2022-03-31"))])

    def test_overlap(self):
        self.assertTrue(overlaps((d("2023-05-08"), d("2023-05-14")), (d("2023-05-01"), d("2023-05-31"))))
        self.assertFalse(overlaps((d("2023-06-01"), d("2023-06-02")), (d("2023-05-01"), d("2023-05-31"))))


if __name__ == "__main__":
    unittest.main()
