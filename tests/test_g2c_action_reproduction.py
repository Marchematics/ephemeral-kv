"""The action parser is the instrument the executable end task is measured with.

A parser regression does not crash anything: it silently turns "the same action" into "a different
action" and moves every rate in the receipt.  These tests pin the two shapes the corpus actually
records (a SWE-agent fenced call and a JSON tool call) plus the failure modes that cost a round
each while the parser was being written - an escaped-quote truncation that made a shell command's
head `{"command":`, and a checkout directory counted as a file target, which would have made every
`cd <checkout> && ...` action "match" on the directory both sides share.
"""

from __future__ import annotations

from benchmarks.g2c_action_reproduction import parse_action, score_action, tool_family
from benchmarks.g2d_action_turns import RETRIEVAL_COMPILER, action_turns


def test_fenced_editor_call():
    action = parse_action("```\nstr_replace_editor view /testbed/funcy/decorators.py\n```")
    assert action["tool"] == "editor"
    assert action["verb"] == "view"
    assert action["target"] == "/testbed/funcy/decorators.py"


def test_editor_arguments_are_not_the_target():
    action = parse_action("```\nstr_replace_editor str_replace /a/b.py --old_str 'x' --new_str 'y'\n```")
    assert (action["tool"], action["verb"], action["target"]) == ("editor", "str_replace", "/a/b.py")


def test_json_tool_call_before_the_name():
    text = ('<assistant>\nPerfect! Let me run the git diff:\n\n\n'
            '[{"function": {"arguments": "{\\"command\\": \\"cd /workspace/p__p__5.19 && '
            'python tests/test_io.py\\"}", "name": "execute_bash"}, "id": "x"}]')
    action = parse_action(text)
    # the escaped-quote truncation showed up here: the head used to parse as `{"command":`
    assert action["tool"] == "bash"
    assert action["raw_tool"] == "execute_bash"
    assert action["target"] == "tests/test_io.py"


def test_checkout_directory_is_not_a_file_target():
    action = parse_action('<assistant>\n[{"function": {"arguments": '
                          '"{\\"command\\": \\"cd /workspace/pandas__pandas__5.19 && git diff\\"}", '
                          '"name": "execute_bash"}]')
    assert action["tool"] == "bash"
    assert action["target"] is None


def test_submit_and_prose():
    assert parse_action("```\nsubmit\n```")["tool"] == "submit"
    assert parse_action("No action here, just prose.") is None


def test_prose_before_a_fence_still_parses():
    action = parse_action("<assistant>\nLet me look:\n```\ncat /testbed/setup.py\n```\n")
    assert (action["tool"], action["target"]) == ("bash", "/testbed/setup.py")


def test_truncated_json_payload_keeps_the_tool():
    # continuations are cut at a token limit, so the payload is often unterminated
    action = parse_action('<assistant>\n[{"name": "str_replace_editor", "arguments": '
                          '"{\\"command\\": \\"str_rep')
    assert action is not None and action["tool"] == "editor"


def test_tool_families_do_not_collapse():
    assert tool_family("execute_bash") == tool_family("bash") == "bash"
    assert tool_family("str_replace_editor") == "editor"
    assert tool_family("execute_bash") != tool_family("str_replace_editor")


def test_score_action_requires_the_same_file():
    recorded = parse_action("```\nstr_replace_editor str_replace /a/b.py --old_str 'x'\n```")
    same = score_action("```\nstr_replace_editor str_replace /a/b.py --old_str 'y'\n```", recorded)
    other = score_action("```\nstr_replace_editor str_replace /a/c.py --old_str 'y'\n```", recorded)
    wrong_tool = score_action("```\ngit diff /a/b.py\n```", recorded)
    assert same["tool"] and same["target"] and same["exact"]
    assert other["tool"] and not other["target"] and not other["exact"]
    assert not wrong_tool["tool"] and not wrong_tool["exact"]


def test_suffix_paths_match():
    recorded = parse_action("```\nstr_replace_editor view /workspace/p__p__5.19/src/x.py\n```")
    generated = parse_action("```\nstr_replace_editor view src/x.py\n```")
    assert score_action("```\nstr_replace_editor view src/x.py\n```", recorded)["target"]
    assert generated["target"] == "src/x.py"


class _Example:
    def __init__(self, target, history):
        self.target = target
        self.history_tokens_estimate = history
        self.full_context = ""
        self.active_context = ""
        self.active_tokens_estimate = 0


def _examples(targets_histories):
    return [_Example(t, h) for t, h in targets_histories]


def test_action_turns_skips_the_submission_and_non_actions():
    """The submission and the prose turn are excluded for what they are, not for where they sit."""
    examples = _examples([
        ("prose only", 100),
        ("```\nstr_replace_editor view /a/b.py\n```", 200),
        ("```\nstr_replace_editor view /a/c.py\n```", 300),
        ("```\nsubmit\n```", 400),
    ])
    picked = action_turns(examples, per_session=2)
    assert [index for index, _, _ in picked] == [1, 2]


def test_history_cap_is_applied_before_the_spread():
    """Filtering after the spread would silently shrink the sample instead of re-spreading it.

    With three eligible turns and two slots, spreading first and filtering afterwards keeps the
    first pick and drops the second, so the session contributes one row where two were asked for.
    """
    six = [("```\nstr_replace_editor view /a/%d.py\n```" % i, i * 100) for i in range(1, 7)]
    assert len(action_turns(_examples(six), per_session=2, max_history_tokens=300)) == 2
    # and the two picks are spread over the eligible range rather than both at the front
    picked = action_turns(_examples(six), per_session=2, max_history_tokens=300)
    assert [e.history_tokens_estimate for _, e, _ in picked] == [100, 300]


def test_retrieval_arm_is_not_the_shipped_view():
    """The comparison arm has to differ in the way the gate names: no window, no consolidation."""
    assert RETRIEVAL_COMPILER["consolidate"] is False
    assert RETRIEVAL_COMPILER["tail_tokens"] == 0
    assert RETRIEVAL_COMPILER["recency_spans"] == 0
    assert RETRIEVAL_COMPILER["dedup"] is False and RETRIEVAL_COMPILER["snippet"] is False
