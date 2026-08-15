from pathlib import Path

from src.verification.requirements import (
    IssueLinkPreflight,
    RequirementAssessment,
    VerificationStatus,
    extract_acceptance_criteria,
    parse_closing_issue_references,
    render_advisory_report,
    resolve_issue_link,
)


def test_closing_issue_references_accept_standard_same_repository_forms() -> None:
    body = """\
Closes #12, owner/repo#34 and https://github.com/owner/repo/issues/56
Fixes other/project#78
Resolves https://github.com/other/project/issues/90
"""

    assert parse_closing_issue_references(body, "owner", "repo") == (12, 34, 56)


def test_relationship_link_is_preferred_over_pr_body_fallback() -> None:
    link = resolve_issue_link(
        [21],
        pr_title="Fixes #99",
        pr_body="Closes #22",
        owner="owner",
        repository="repo",
    )

    assert link.issue_number == 21
    assert link.status is VerificationStatus.PASS


def test_report_preflight_passes_fully_linked_satisfiable_context() -> None:
    link = resolve_issue_link([21], pr_title="", pr_body="", owner="owner", repository="repo")
    report = render_advisory_report(
        link,
        [
            RequirementAssessment(
                "Publish one idempotent advisory comment.",
                VerificationStatus.PASS,
                "The marker lookup updates an existing comment.",
                "The focused test passed.",
                "None.",
            )
        ],
    )

    assert "Overall verdict: **PASS**" in report
    assert "[PASS] Publish one idempotent advisory comment." in report


def test_no_or_ambiguous_link_is_unverified() -> None:
    no_link = resolve_issue_link(
        [],
        pr_title="Update docs",
        pr_body="Fixes outside/repo#42",
        owner="owner",
        repository="repo",
    )
    ambiguous_link = resolve_issue_link(
        [], pr_title="", pr_body="Closes #2 and #3", owner="owner", repository="repo"
    )

    assert no_link.status is VerificationStatus.UNVERIFIED
    assert ambiguous_link.status is VerificationStatus.UNVERIFIED
    assert "multiple" in ambiguous_link.reason.lower()


def test_report_preflight_represents_satisfied_and_missing_requirements() -> None:
    link = IssueLinkPreflight(7, VerificationStatus.PASS, "Linked issue #7.")
    criteria = extract_acceptance_criteria(
        """\
## Acceptance criteria

- [ ] Add deterministic link handling.
- [x] Publish one idempotent advisory comment.

## Scope
"""
    )
    report = render_advisory_report(
        link,
        [
            RequirementAssessment(
                criteria[0],
                VerificationStatus.PASS,
                "Helper test covers the reference.",
                "Focused test passed.",
                "None.",
            ),
            RequirementAssessment(
                criteria[1],
                VerificationStatus.FAIL,
                "No implementation evidence found.",
                "No CI evidence found.",
                "None.",
            ),
        ],
    )

    assert criteria == (
        "Add deterministic link handling.",
        "Publish one idempotent advisory comment.",
    )
    assert "Overall verdict: **FAIL**" in report
    assert "### 2. [FAIL] Publish one idempotent advisory comment." in report


def test_report_preflight_without_reliable_issue_is_unverified() -> None:
    report = render_advisory_report(
        IssueLinkPreflight(None, VerificationStatus.UNVERIFIED, "No reliable linked issue."),
    )

    assert "Overall verdict: **UNVERIFIED**" in report
    assert "No reliable linked issue." in report


def test_workflow_has_post_ci_trigger_and_secret_isolated_permissions() -> None:
    workflow = Path(".github/workflows/requirement-verification.yml").read_text()
    verify_job = workflow.split("  verify:\n", maxsplit=1)[1].split("  publish:\n", maxsplit=1)[0]
    publish_job = workflow.split("  publish:\n", maxsplit=1)[1]

    assert "workflow_run:" in workflow
    assert "workflows: [CI]" in workflow
    assert "types: [completed]" in workflow
    assert "head.repo?.full_name !== `${owner}/${repo}`" in workflow
    assert "openai-api-key: ${{ secrets.OPENAI_API_KEY }}" in verify_job
    assert "issues: write" not in verify_job
    assert "permissions:\n      issues: write" in publish_job
    assert "contents: read" not in publish_job
    assert "pull-requests: read" not in publish_job
    assert "actions: read" not in publish_job
