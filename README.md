# ai-code-reviewer

An AI-powered GitHub PR review bot. On every pull request event, a GitHub Actions
workflow fetches the PR diff, sends it to Gemini (`gemini-2.5-flash`, free tier)
for structured review, and posts the results back as **inline, line-level
comments** on the PR via the GitHub review API.

## How it works

1. **Trigger**: `.github/workflows/review.yml` runs on `pull_request` (`opened`,
   `synchronize`).
2. **Diff extraction**: `review_pr.py` pulls each changed file's patch via PyGithub
   and parses the unified diff into per-line records (`parse_patch`), tracking which
   new-file line numbers are actually commentable (added/context lines present in
   the diff).
3. **Structured review**: The annotated diff (each line prefixed with its real
   line number) is sent to Gemini with a forced function call
   (`submit_code_review`, `tool_config` mode `ANY`) so the model returns a strict
   JSON shape — a summary plus a list of `{path, line, severity, comment}` —
   instead of free-form prose.
4. **Validation**: Every proposed comment is checked against the actual
   commentable lines for that file. Comments pointing at lines outside the diff
   (a common failure mode for freeform LLM output) are dropped rather than sent to
   GitHub, where they'd cause a 422.
5. **Delivery**: `pr.create_review(...)` posts one PR review with all valid
   comments attached to their exact file/line, plus a short overall summary.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` for local runs (not needed in CI):

```
GEMINI_API_KEY=...
GITHUB_TOKEN=ghp_...
```

Get a free `GEMINI_API_KEY` at [Google AI Studio](https://aistudio.google.com/apikey) —
no billing account required. In the GitHub repo, add `GEMINI_API_KEY` as a
repository secret (Settings → Secrets and variables → Actions). `GITHUB_TOKEN`
is provided automatically by Actions.

## Running locally

```bash
python review_pr.py <owner/repo> <pr_number>
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover the diff-parsing/line-mapping logic and the Gemini function-call
response handling (mocked — no API key required).
