"""Tests for deterministic fitness scoring helpers."""

import dspy

from evolution.core.fitness import _meaningful_tokens, _parse_score, skill_fitness_metric


def _score(expected_behavior: str, output: str) -> float:
    example = dspy.Example(expected_behavior=expected_behavior, task_input="do the task")
    prediction = dspy.Prediction(output=output)
    return skill_fitness_metric(example, prediction)


class TestMeaningfulTokens:
    def test_normalizes_case_and_punctuation(self):
        assert _meaningful_tokens("Group, GROUP topics!") == {"group", "topics"}

    def test_filters_short_words_and_stopwords(self):
        assert _meaningful_tokens("to be in the API and UI") == {"api"}


class TestSkillFitnessMetric:
    def test_empty_output_scores_zero(self):
        assert _score("group by topic", "   ") == 0.0

    def test_non_empty_output_without_rubric_keeps_neutral_score(self):
        assert _score("", "I can help with that.") == 0.5

    def test_complete_keyword_coverage_scores_high(self):
        score = _score(
            "group messages by topic and summarize outliers",
            "I will group the messages by topic, then summarize outliers.",
        )
        assert score > 0.85

    def test_partial_keyword_coverage_scores_lower_than_complete(self):
        complete = _score(
            "group messages by topic and summarize outliers",
            "Group messages by topic and summarize outliers.",
        )
        partial = _score(
            "group messages by topic and summarize outliers",
            "Group messages by topic.",
        )

        assert partial < complete
        assert 0.2 < partial < 0.8

    def test_verbose_keyword_stuffing_is_penalized_by_precision(self):
        focused = _score(
            "group messages by topic",
            "Group messages by topic.",
        )
        verbose = _score(
            "group messages by topic",
            "Group messages by topic, then discuss unrelated deployment tests, "
            "authentication, billing, logging, dashboards, and migrations.",
        )

        assert verbose < focused


class TestParseScore:
    def test_clamps_numeric_values(self):
        assert _parse_score(1.5) == 1.0
        assert _parse_score(-0.5) == 0.0

    def test_defaults_to_neutral_for_unparseable_values(self):
        assert _parse_score("not a score") == 0.5
