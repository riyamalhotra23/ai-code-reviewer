import os
import sys
from github import Github
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert code reviewer. When given a pull request diff, provide a thorough, constructive review covering:

1. **Summary**: A brief overview of what the PR changes
2. **Issues**: Bugs, security vulnerabilities, or correctness problems (if any)
3. **Suggestions**: Improvements to readability, performance, or maintainability
4. **Positives**: Things done well in this PR

Be specific, reference line numbers or code snippets where applicable, and keep feedback actionable."""


def get_pr_diff(repo_name: str, pr_number: int) -> str:
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    diff_parts = []
    for f in pr.get_files():
        diff_parts.append(f"### {f.filename} ({f.status})")
        if f.patch:
            diff_parts.append(f.patch)
        diff_parts.append("")

    return "\n".join(diff_parts)


def review_with_claude(diff: str) -> str:
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Please review this pull request diff:\n\n{diff}",
            }
        ],
    )
    return response.content[0].text


def post_pr_comment(repo_name: str, pr_number: int, comment: str) -> None:
    github_token = os.environ.get("GITHUB_TOKEN")
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(f"## AI Code Review\n\n{comment}")
    print(f"Review posted to PR #{pr_number} in {repo_name}")


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

    print(f"Fetching diff for PR #{pr_number} in {repo_name}...")
    diff = get_pr_diff(repo_name, pr_number)

    if not diff.strip():
        print("No diff found for this PR.", file=sys.stderr)
        sys.exit(1)

    print("Sending diff to Claude for review...")
    review = review_with_claude(diff)

    print("Posting review comment...")
    post_pr_comment(repo_name, pr_number, review)


if __name__ == "__main__":
    main()
