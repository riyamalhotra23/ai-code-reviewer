import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from review_pr import annotate_patch, parse_patch, review_with_claude, validate_comments

NEW_FILE_PATCH = """@@ -0,0 +1,4 @@
+def add(a, b):
+    return a + b
+
+print(add(5, 3))"""

MIXED_PATCH = """@@ -1,5 +1,6 @@
 def add(a, b):
-    return a+b
+    return a + b
+
 def sub(a, b):
     return a - b
"""


class TestParsePatch(unittest.TestCase):
    def test_new_file_all_added_lines(self):
        lines = parse_patch(NEW_FILE_PATCH)
        self.assertEqual([dl.kind for dl in lines], ["add", "add", "add", "add"])
        self.assertEqual([dl.new_line for dl in lines], [1, 2, 3, 4])

    def test_mixed_context_add_del(self):
        lines = parse_patch(MIXED_PATCH)
        kinds = [dl.kind for dl in lines]
        self.assertEqual(kinds, ["context", "del", "add", "add", "context", "context"])
        new_lines = [dl.new_line for dl in lines if dl.kind != "del"]
        self.assertEqual(new_lines, [1, 2, 3, 4, 5])

    def test_deleted_lines_have_no_new_line_number(self):
        lines = parse_patch(MIXED_PATCH)
        deleted = [dl for dl in lines if dl.kind == "del"]
        self.assertEqual(len(deleted), 1)
        self.assertIsNone(deleted[0].new_line)
        self.assertEqual(deleted[0].old_line, 2)

    def test_annotate_patch_marks_commentable_lines(self):
        lines = parse_patch(NEW_FILE_PATCH)
        rendered = annotate_patch(lines)
        self.assertIn("1 + def add(a, b):", rendered)
        self.assertIn("4 + print(add(5, 3))", rendered)


class TestValidateComments(unittest.TestCase):
    def setUp(self):
        self.commentable = {"calculator.py": {1, 2, 3, 4}}

    def test_valid_comment_kept(self):
        comments = [{"path": "calculator.py", "line": 2, "severity": "bug", "comment": "x"}]
        valid, dropped = validate_comments(comments, self.commentable)
        self.assertEqual(valid, comments)
        self.assertEqual(dropped, [])

    def test_out_of_range_line_dropped(self):
        comments = [{"path": "calculator.py", "line": 99, "severity": "bug", "comment": "x"}]
        valid, dropped = validate_comments(comments, self.commentable)
        self.assertEqual(valid, [])
        self.assertEqual(dropped, comments)

    def test_unknown_path_dropped(self):
        comments = [{"path": "nope.py", "line": 1, "severity": "bug", "comment": "x"}]
        valid, dropped = validate_comments(comments, self.commentable)
        self.assertEqual(valid, [])
        self.assertEqual(dropped, comments)


class TestReviewWithClaude(unittest.TestCase):
    def test_extracts_tool_use_input(self):
        expected_input = {
            "summary": "Adds an add() helper.",
            "comments": [
                {"path": "calculator.py", "line": 2, "severity": "nit", "comment": "Add a docstring."}
            ],
        }
        fake_block = SimpleNamespace(type="tool_use", name="submit_code_review", input=expected_input)
        fake_response = SimpleNamespace(content=[fake_block])

        with patch("review_pr.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = fake_response
            result = review_with_claude("### calculator.py (added)\n1 + def add(a, b):")

        self.assertEqual(result, expected_input)

    def test_raises_if_no_tool_use_block(self):
        fake_response = SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")])
        with patch("review_pr.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.return_value = fake_response
            with self.assertRaises(RuntimeError):
                review_with_claude("some diff")


if __name__ == "__main__":
    unittest.main()
