"""Pure helpers used to preflight advisory requirement-verification reports."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class VerificationStatus(str, Enum):
    """Allowed statuses in an advisory requirement-verification report."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class IssueLinkPreflight:
    """The deterministic result of resolving the issue a PR closes."""

    issue_number: int | None
    status: VerificationStatus
    reason: str


@dataclass(frozen=True)
class RequirementAssessment:
    """An evidence-backed assessment for one verbatim issue criterion."""

    criterion: str
    status: VerificationStatus
    implementation_evidence: str
    test_ci_evidence: str
    assumptions: str


_REFERENCE = r"(?:#\d+|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+|https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+)"
_CLOSING_CLAUSE = re.compile(
    rf"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*"
    rf"(?P<references>{_REFERENCE}(?:\s*(?:,|and)\s*{_REFERENCE})*)"
)
_REFERENCE_PART = re.compile(
    r"^(?:#(?P<local>\d+)|(?P<qualified_owner>[A-Za-z0-9_.-]+)/"
    r"(?P<qualified_repo>[A-Za-z0-9_.-]+)#(?P<qualified_number>\d+)|"
    r"https?://github\.com/(?P<url_owner>[A-Za-z0-9_.-]+)/"
    r"(?P<url_repo>[A-Za-z0-9_.-]+)/issues/(?P<url_number>\d+))$",
    re.IGNORECASE,
)
_CRITERIA_HEADING = re.compile(r"(?im)^#{1,6}\s+acceptance criteria\s*$")
_NEXT_HEADING = re.compile(r"(?m)^#{1,6}\s+")
_CHECKBOX_ITEM = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s+(?P<criterion>.+?)\s*$")


def parse_closing_issue_references(text: str, owner: str, repository: str) -> tuple[int, ...]:
    """Return unique same-repository issue numbers from GitHub closing syntax.

    References that name another repository are deliberately ignored. A caller
    must not infer that a cross-repository issue is the target of this verifier.
    """

    normalized_owner = owner.casefold()
    normalized_repository = repository.casefold()
    numbers: set[int] = set()
    for clause in _CLOSING_CLAUSE.finditer(text):
        for raw_reference in re.split(r"\s*(?:,|and)\s*", clause.group("references")):
            match = _REFERENCE_PART.fullmatch(raw_reference)
            if match is None:
                continue
            if match.group("local") is not None:
                numbers.add(int(match.group("local")))
                continue

            reference_owner = match.group("qualified_owner") or match.group("url_owner")
            reference_repository = match.group("qualified_repo") or match.group("url_repo")
            reference_number = match.group("qualified_number") or match.group("url_number")
            if (
                reference_owner is not None
                and reference_repository is not None
                and reference_number is not None
                and reference_owner.casefold() == normalized_owner
                and reference_repository.casefold() == normalized_repository
            ):
                numbers.add(int(reference_number))
    return tuple(sorted(numbers))


def resolve_issue_link(
    closing_issue_numbers: Iterable[int],
    *,
    pr_title: str,
    pr_body: str,
    owner: str,
    repository: str,
) -> IssueLinkPreflight:
    """Resolve one issue, preferring GitHub's closing-issue relationship."""

    relationship_numbers = tuple(sorted(set(closing_issue_numbers)))
    if len(relationship_numbers) == 1:
        return IssueLinkPreflight(
            issue_number=relationship_numbers[0],
            status=VerificationStatus.PASS,
            reason="GitHub closing-issue relationship resolved exactly one same-repository issue.",
        )
    if len(relationship_numbers) > 1:
        return IssueLinkPreflight(
            issue_number=None,
            status=VerificationStatus.UNVERIFIED,
            reason="GitHub closing-issue relationship resolved multiple same-repository issues.",
        )

    fallback_numbers = set(parse_closing_issue_references(pr_title, owner, repository))
    fallback_numbers.update(parse_closing_issue_references(pr_body, owner, repository))
    if len(fallback_numbers) == 1:
        return IssueLinkPreflight(
            issue_number=next(iter(fallback_numbers)),
            status=VerificationStatus.PASS,
            reason="One same-repository closing reference was parsed from the pull request metadata.",
        )
    if len(fallback_numbers) > 1:
        return IssueLinkPreflight(
            issue_number=None,
            status=VerificationStatus.UNVERIFIED,
            reason="Multiple same-repository closing references were parsed from the pull request metadata.",
        )
    return IssueLinkPreflight(
        issue_number=None,
        status=VerificationStatus.UNVERIFIED,
        reason="No same-repository closing issue could be resolved.",
    )


def extract_acceptance_criteria(issue_body: str) -> tuple[str, ...]:
    """Extract checked or unchecked Markdown checklist criteria from the named section."""

    heading = _CRITERIA_HEADING.search(issue_body)
    if heading is None:
        return ()
    section_start = heading.end()
    next_heading = _NEXT_HEADING.search(issue_body, section_start)
    section = issue_body[section_start : next_heading.start() if next_heading else None]
    criteria = [
        item.group("criterion").strip()
        for line in section.splitlines()
        if (item := _CHECKBOX_ITEM.match(line)) is not None and item.group("criterion").strip()
    ]
    return tuple(criteria)


def render_advisory_report(
    link: IssueLinkPreflight,
    assessments: Iterable[RequirementAssessment] = (),
) -> str:
    """Render a deterministic preflight/fallback report for an advisory PR comment."""

    assessment_list = tuple(assessments)
    if link.status is VerificationStatus.UNVERIFIED:
        return "\n".join(
            [
                "## Requirement verification",
                "",
                "Overall verdict: **UNVERIFIED**",
                "",
                link.reason,
            ]
        )

    overall = (
        VerificationStatus.PASS
        if assessment_list
        and all(item.status is VerificationStatus.PASS for item in assessment_list)
        else VerificationStatus.FAIL
    )
    lines = ["## Requirement verification", "", f"Overall verdict: **{overall.value}**"]
    for index, assessment in enumerate(assessment_list, start=1):
        lines.extend(
            [
                "",
                f"### {index}. [{assessment.status.value}] {assessment.criterion}",
                f"- Implementation evidence: {assessment.implementation_evidence}",
                f"- Test/CI evidence: {assessment.test_ci_evidence}",
                f"- Assumptions: {assessment.assumptions}",
            ]
        )
    return "\n".join(lines)
