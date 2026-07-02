import json
import os
import re
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from github import Github
from google import genai
from google.genai import types

load_dotenv()

# Free-tier Gemini model. Check https://aistudio.google.com/rate-limit for current
# free-tier availability if this model stops working.
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an expert code reviewer embedded in a CI pipeline. You will be shown a \
pull request diff, file by file. Each shown line is prefixed with its line number in the NEW \
version of the file (removed lines have no number and are marked "-"; unnumbered lines cannot be \
commented on).

Call the `submit_code_review` tool exactly once with your findings. Rules:
- Only comment on lines that show a line number in the diff you were given.
- Every inline comment must point at the single most relevant line for that issue.
- Focus on real bugs, security vulnerabilities, correctness problems, and meaningful \
maintainability/performance issues. Do not nitpick style that a linter would already catch.
- Skip praise-only comments; only leave a comment if there's something actionable to say.
- Keep each comment to 1-3 sentences. Be specific and reference exact identifiers.
- `summary` should be 2-4 sentences describing what the PR does and your overall assessment.
- If the diff has nothing worth flagging, return an empty `comments` array.
"""

REVIEW_TOOL_NAME = "submit_code_review"

REVIEW_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Overall summary and assessment of the PR (2-4 sentences).",
        },
        "comments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path exactly as shown in the diff header.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "New-file line number this comment applies to.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["bug", "security", "performance", "suggestion", "nit"],
                    },
                    "comment": {
                        "type": "string",
                        "description": "The actionable feedback for this line.",
                    },
                },
                "required": ["path", "line", "severity", "comment"],
            },
        },
    },
    "required": ["summary", "comments"],
}

SEVERITY_LABELS = {
    "bug": "🐛 Bug",
    "security": "🔒 Security",
    "performance": "⚡ Performance",
    "suggestion": "💡 Suggestion",
    "nit": "✏️ Nit",
}


@dataclass
class DiffLine:
    new_line: int | None
    old_line: int | None
    kind: str  # "add", "del", "context"
    content: str


def parse_patch(patch: str) -> list[DiffLine]:
    """Parse a unified diff patch (GitHub's `File.patch` format) into per-line records."""
    lines: list[DiffLine] = []
    old_ln = new_ln = 0
    hunk_re = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for raw in patch.splitlines():
        m = hunk_re.match(raw)
        if m:
            old_ln, new_ln = int(m.group(1)), int(m.group(2))
            continue
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        if raw.startswith("+"):
            lines.append(DiffLine(new_ln, None, "add", raw[1:]))
            new_ln += 1
        elif raw.startswith("-"):
            lines.append(DiffLine(None, old_ln, "del", raw[1:]))
            old_ln += 1
        else:
            content = raw[1:] if raw.startswith(" ") else raw
            lines.append(DiffLine(new_ln, old_ln, "context", content))
            old_ln += 1
            new_ln += 1

    return lines


def annotate_patch(lines: list[DiffLine]) -> str:
    """Render parsed diff lines with new-file line numbers so the model can cite valid lines."""
    out = []
    for dl in lines:
        if dl.kind == "add":
            out.append(f"{dl.new_line:>5} + {dl.content}")
        elif dl.kind == "del":
            out.append(f"    - - {dl.content}")
        else:
            out.append(f"{dl.new_line:>5}   {dl.content}")
    return "\n".join(out)


def build_review_prompt(repo, pr) -> tuple[str, dict[str, set[int]]]:
    """Return (annotated diff text, {path: set of commentable new-file line numbers})."""
    sections = []
    commentable: dict[str, set[int]] = {}

    for f in pr.get_files():
        if not f.patch:
            sections.append(f"### {f.filename} ({f.status}) — binary or no textual diff\n")
            continue
        parsed = parse_patch(f.patch)
        commentable[f.filename] = {
            dl.new_line for dl in parsed if dl.kind in ("add", "context") and dl.new_line is not None
        }
        sections.append(f"### {f.filename} ({f.status})\n{annotate_patch(parsed)}\n")

    return "\n".join(sections), commentable


def review_with_gemini(diff_text: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=REVIEW_TOOL_NAME,
                description="Submit a structured code review for the pull request diff.",
                parameters_json_schema=REVIEW_TOOL_SCHEMA,
            )
        ]
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Review this pull request diff:\n\n{diff_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=8192,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[REVIEW_TOOL_NAME],
                )
            ),
        ),
    )

    for call in response.function_calls or []:
        if call.name == REVIEW_TOOL_NAME:
            return call.args

    raise RuntimeError("Gemini did not return a submit_code_review function call")


def validate_comments(raw_comments: list[dict], commentable: dict[str, set[int]]) -> tuple[list[dict], list[dict]]:
    """Split model comments into (valid, dropped) based on which lines actually exist in the diff."""
    valid, dropped = [], []
    for c in raw_comments:
        path, line = c.get("path"), c.get("line")
        if path in commentable and isinstance(line, int) and line in commentable[path]:
            valid.append(c)
        else:
            dropped.append(c)
    return valid, dropped


def post_inline_review(pr, summary: str, comments: list[dict], dropped: list[dict]) -> None:
    body_parts = [f"## 🤖 AI Code Review\n\n{summary}"]
    if dropped:
        body_parts.append(
            f"\n_Note: {len(dropped)} comment(s) referenced lines outside the diff and were omitted._"
        )

    review_comments = [
        {
            "path": c["path"],
            "line": c["line"],
            "side": "RIGHT",
            "body": f"**{SEVERITY_LABELS.get(c['severity'], c['severity'])}**\n\n{c['comment']}",
        }
        for c in comments
    ]

    pr.create_review(body="\n".join(body_parts), event="COMMENT", comments=review_comments)
    print(f"Posted review with {len(review_comments)} inline comment(s).")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <repo_name> <pr_number>")
        print(f"  Example: {sys.argv[0]} owner/repo 42")
        sys.exit(1)

    repo_name = sys.argv[1]
    try:
        pr_number = int(sys.argv[2])
    except ValueError:
        print(f"Error: pr_number must be an integer, got '{sys.argv[2]}'", file=sys.stderr)
        sys.exit(1)

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    print(f"Building diff for PR #{pr_number} in {repo_name}...")
    diff_text, commentable = build_review_prompt(repo, pr)

    if not diff_text.strip():
        print("No diff found for this PR.", file=sys.stderr)
        sys.exit(1)

    print("Sending diff to Gemini for structured review...")
    result = review_with_gemini(diff_text)

    valid, dropped = validate_comments(result.get("comments", []), commentable)
    if dropped:
        print(f"Warning: dropped {len(dropped)} comment(s) with invalid path/line: {json.dumps(dropped)}")

    print("Posting inline review...")
    post_inline_review(pr, result.get("summary", ""), valid, dropped)


if __name__ == "__main__":
    main()
