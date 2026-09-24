"""Patch-localization scoring: the label extraction and the suffix matching.

The end-task half of G2 is only as good as (a) recovering the recorded patch's files from a
trajectory and (b) matching a continuation's file mentions to them, since one file can be
spelled `/workspace/<checkout>/pkg/mod.py`, `pkg/mod.py` or `b/pkg/mod.py` in the same
transcript.  These run on CPU, on hand-built transcripts that mirror the real shapes.
"""

from __future__ import annotations

from benchmarks.g2b_patch_localization import (mentioned_files,
                                               patch_files_from_messages,
                                               score_continuation)


def test_files_come_from_tool_output_and_tool_arguments():
    messages = [
        {"role": "assistant", "content": "Let me look at the diff",
         "tool_calls_json": [{"function": {"name": "execute_bash",
                                           "arguments": '{"command": "git diff"}'}}]},
        {"role": "tool", "content": "diff --git a/pkg/mod.py b/pkg/mod.py\nindex 1..2 100644"},
        {"role": "tool", "content": "*** Update File: src/other/thing.py\n"},
        {"role": "assistant",
         "tool_calls_json": [{"function": {"arguments": '{"path": "lib/deep/impl.c"}'}}]},
        {"role": "assistant", "content": "**Modified File**: `docs/readme.md`"},
    ]
    files = patch_files_from_messages(messages)
    assert files == {"pkg/mod.py", "src/other/thing.py", "lib/deep/impl.c", "docs/readme.md"}


def test_assistant_prose_alone_is_not_enough():
    """The real trajectories carry the patch in tool output, not in prose - the first
    version of the extractor scanned assistant content only and found almost nothing."""
    prose_only = [{"role": "assistant", "content": "I changed the parser in src/parse.py"}]
    assert patch_files_from_messages(prose_only) == set()


def test_checkout_prefixes_are_normalised_away():
    messages = [{"role": "tool",
                 "content": "diff --git a/pandas/core/arrays/period.py b/pandas/core/arrays/period.py"},
                {"role": "tool", "content": "*** Update File: /workspace/pandas__pandas__5.19/x/y.py"}]
    files = patch_files_from_messages(messages)
    assert "pandas/core/arrays/period.py" in files
    assert "x/y.py" in files                      # the checkout directory is dropped


def test_score_matches_absolute_and_relative_spellings():
    recorded = {"pandas/core/arrays/period.py"}
    text = "I will update pandas/core/arrays/period.py to handle the new dtype."
    score = score_continuation(text, recorded)
    assert score["recall"] == 1.0 and score["f1"] == 1.0

    text_abs = ("The fix belongs in /workspace/pandas__pandas__5.19/pandas/core/arrays/"
                "period.py")
    assert score_continuation(text_abs, recorded)["recall"] == 1.0


def test_score_penalises_naming_unrelated_files():
    recorded = {"pkg/mod.py"}
    good = score_continuation("edit pkg/mod.py", recorded)
    noisy = score_continuation("edit pkg/mod.py and also tests/other_test.py", recorded)
    assert good["precision"] == 1.0
    assert noisy["precision"] < 1.0 and noisy["recall"] == 1.0


def test_score_is_zero_when_nothing_is_named():
    score = score_continuation("I will investigate the issue further.", {"pkg/mod.py"})
    assert score["f1"] == 0.0 and score["hits"] == 0


def test_mentioned_files_ignores_non_source_tokens():
    assert mentioned_files("run the tests with pytest -q") == set()
    assert mentioned_files("see src/app.py and config.yaml") == {"src/app.py", "config.yaml"}
