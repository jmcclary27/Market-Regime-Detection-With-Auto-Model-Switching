from pathlib import Path


def test_developer_agent_workflow_has_controlled_dispatch_and_isolated_publish_token() -> None:
    workflow = Path(".github/workflows/codex-issue-implementation.yml").read_text(encoding="utf-8")
    preflight = workflow.split("  preflight:\n", maxsplit=1)[1].split("  implement:\n", maxsplit=1)[
        0
    ]
    implementation = workflow.split("  implement:\n", maxsplit=1)[1].split(
        "  publish:\n", maxsplit=1
    )[0]
    publish = workflow.split("  publish:\n", maxsplit=1)[1]

    assert "workflow_dispatch:" in workflow
    assert "issue_number:" in workflow
    assert "Dispatch this workflow from the repository default branch only." in preflight
    assert '!["admin", "maintain"].includes' in preflight
    assert "Input #${issueNumber} is a pull request, not an issue." in preflight
    assert "needs at least one checklist item under ## Acceptance criteria" in preflight
    assert "agent/issue-${issueNumber}-" in preflight
    assert "refusing to overwrite it" in preflight

    assert "openai-api-key: ${{ secrets.OPENAI_API_KEY }}" in implementation
    assert "sandbox: workspace-write" in implementation
    assert "safety-strategy: drop-sudo" in implementation
    assert "CODEX_AGENT_GITHUB_TOKEN" not in implementation
    assert 'docker run --rm market-regime:agent sh -lc "pytest -q"' in implementation
    assert "--entrypoint python market-regime:agent -c" in implementation
    assert "agent-pr-description.md" in implementation

    assert "CODEX_AGENT_GITHUB_TOKEN: ${{ secrets.CODEX_AGENT_GITHUB_TOKEN }}" in publish
    assert 'git push origin "HEAD:refs/heads/$AGENT_BRANCH"' in publish
    assert "--force" not in publish
    assert "draft: false" in publish
    assert "Closes #${issueNumber}" in publish
    assert "pulls.create" in publish
    assert "merges." not in publish
    assert "deploy" not in publish.lower()


def test_developer_agent_prompt_prohibits_mutation_outside_the_issue_branch() -> None:
    prompt = Path(".github/prompts/developer-agent.md").read_text(encoding="utf-8")

    assert "agent-input/issue-context.md" in prompt
    assert "untrusted" in prompt
    assert "agent-pr-description.md" in prompt
    assert "Do not commit, push," in prompt
    assert "open or update a pull request, merge, approve, deploy" in prompt
    assert "directly to the default branch." in prompt
