"""Source documents: a seeded synthetic handbook, a JSONL loader, word-window chunking.

Only pydantic and the standard library are imported here so the load-test
question bank (``rag_load_test.loadtest.questions``) stays cheap to import.

The synthetic corpus describes a fictional company, Northwind Analytics. Each
topic owns a distinctive vocabulary (its own nouns and even its own time unit)
that its sentences repeat and its question reuses in the same word form, and
shares as little as possible with other topics. That keeps retrieval meaningful
even for a bag-of-words embedder such as ``FakeEmbedder`` (16 dims), which
otherwise conflates passages that merely share "days", "employees" or the
company name.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ..contracts.models import MetadataValue, Passage


class SourceDocument(BaseModel):
    """A whole document before chunking.

    Example::

        doc = SourceDocument(id="hb-1", title="Handbook", text="...", metadata={})
    """

    id: str
    title: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


# (topic_slug, title_template, sentence_templates). Every sentence carries at
# least one numeric slot ({n}, {days}, {amount}, {hours}, {pct}) filled from the
# seeded RNG so documents of the same topic differ.
SYNTHETIC_TOPICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "vacation-policy",
        "Northwind Analytics Vacation Policy (revision {n})",
        (
            "Full-time employees at Northwind Analytics accrue {days} paid vacation "
            "days each year, and part-time employees accrue vacation pro rata.",
            "Vacation days must be requested at least {n} days ahead so the team "
            "can plan cover for the vacation.",
            "Employees may carry up to {n} unused paid vacation days into the next "
            "year; any further vacation days are forfeited.",
            "New employees accrue vacation from their first day and may take paid "
            "vacation days after {days} days of service.",
            "A vacation longer than {n} consecutive days needs sign-off from the "
            "department head before the days are booked.",
        ),
    ),
    (
        "expense-reimbursement",
        "Northwind Analytics Expense Reimbursement Guide (revision {n})",
        (
            "Northwind Analytics pays reimbursement for any business expense claim "
            "filed through the claim portal within {n} weeks, and each claim lists "
            "the dollars spent.",
            "An itemized receipt is required for a claim over {amount} dollars; a "
            "claim under {amount} dollars needs only a short note instead of a "
            "receipt.",
            "Reimbursement lands with the next payroll, usually {n} weeks after the "
            "manager approves the expense claim, and the dollars arrive as a refund "
            "on the payslip.",
            "Meals qualify for reimbursement up to {amount} dollars per meal; "
            "alcohol is never a reimbursable expense, so no receipt for alcohol is "
            "accepted.",
            "A claim with a missing receipt is sent back, and repeat offenders lose "
            "reimbursement after {n} rejected claims.",
        ),
    ),
    (
        "remote-work",
        "Northwind Analytics Remote Work Policy (revision {n})",
        (
            "Employees at Northwind Analytics may work remotely from home up to {n} "
            "times per week with their manager's agreement.",
            "Remote work from home requires a secure home network and a company "
            "computer; personal devices are blocked after {n} weeks.",
            "Each team picks one shared office day per week; employees who work "
            "remotely still join the office for at least {n} weeks per quarter.",
            "Employees who work remotely from home in another country for more than "
            "{n} weeks need sign-off from HR and legal to keep working remotely.",
            "The remote work stipend is worth {amount} per year of home office "
            "furniture for employees who work remotely at least half the week.",
        ),
    ),
    (
        "security-training",
        "Northwind Analytics Security Training Requirements (revision {n})",
        (
            "Every quarter, all staff at Northwind Analytics finish the mandatory "
            "security module within {n} weeks of it opening; new staff finish it "
            "in their first quarter.",
            "The security training module covers phishing, password hygiene and "
            "safe file handling, and takes about {hours} hours to finish.",
            "Staff who fail one of the monthly phishing tests finish an extra "
            "security module within {hours} hours of the phishing tests results.",
            "Security module completion stays above {pct} percent per team each "
            "quarter, or the whole team repeats the phishing tests.",
            "Contractors finish the same mandatory security module and phishing "
            "tests, and lose system access once the module is {n} weeks overdue.",
        ),
    ),
    (
        "onboarding",
        "Northwind Analytics Onboarding Checklist (revision {n})",
        (
            "Onboarding at Northwind Analytics lasts {n} weeks and follows an "
            "onboarding checklist owned by the hire's manager.",
            "The onboarding checklist covers accounts, badge access, tooling setup "
            "and {n} introductory meetings for the hire.",
            "Every new hire is paired with an onboarding buddy within {hours} hours; "
            "the buddy stays assigned for {n} weeks.",
            "The hire completes the onboarding checklist in the people portal, and "
            "the buddy checks any checklist task still open after {n} weeks.",
            "An onboarding survey goes to every hire after {n} weeks so the buddy "
            "programme and the checklist keep improving for the next hire.",
        ),
    ),
    (
        "incident-response",
        "Northwind Analytics Incident Response Runbook (revision {n})",
        (
            "The on-call engineers must respond to a production incident alert "
            "within {n} minutes of the alert firing.",
            "Northwind Analytics classifies every incident by severity; a "
            "severity-one production incident pages {n} engineers, and the on-call "
            "lead coordinates the fix.",
            "Incident handling follows one sequence: acknowledge the incident "
            "alert, mitigate the outage, post an update every {n} minutes, then "
            "resolve.",
            "A blameless incident write-up is published within {hours} hours of "
            "the outage and shared with all engineers.",
            "Customers hear about a production incident lasting more than {hours} "
            "hours through the public incident page, which the on-call engineers "
            "must keep current.",
        ),
    ),
    (
        "data-retention",
        "Northwind Analytics Data Retention Policy (revision {n})",
        (
            "The data retention policy at Northwind Analytics is to retain customer "
            "data for {n} months after a contract ends and not a month longer.",
            "Encrypted backups of customer data are retained for {n} months, then "
            "the backups are rotated and the old backups destroyed.",
            "Application logs are retained for {n} months, while audit logs must be "
            "retained for {n} years by regulation.",
            "Customers may ask to have their customer data deleted early; the "
            "retention team removes the data and its backups within {n} weeks.",
            "Any exception to how long we retain customer data requires legal "
            "sign-off and is re-checked every {n} months.",
        ),
    ),
    (
        "travel-booking",
        "Northwind Analytics Travel Booking Policy (revision {n})",
        (
            "Staff get approval for every business trip via the Northwind Analytics "
            "travel portal at least {n} weeks before the flight departs; the "
            "airfare is booked there too.",
            "A flight over {amount} dollars requires manager approval via the "
            "portal before the airfare is booked; the approval is attached to the "
            "trip.",
            "Standard fares are the default for any flight under {hours} hours; a "
            "longer flight may be booked at a premium fare with the same approval.",
            "Hotel rates are capped at {amount} dollars per night for a business "
            "trip unless the travel desk documents an exception.",
            "Trip insurance covers a business trip of up to {n} weeks; a longer "
            "trip requires approval plus a separate policy booked before the "
            "flight.",
        ),
    ),
    (
        "performance-reviews",
        "Northwind Analytics Performance Review Process (revision {n})",
        (
            "A performance review is held twice a year at Northwind Analytics; a "
            "self-evaluation is due on day {n} of the review cycle.",
            "Each performance review produces a review score on a scale from 1 to "
            "{n}; every score on the scale is backed by written examples.",
            "Peer evaluation from at least {n} colleagues is gathered before the "
            "performance review score is decided.",
            "Score calibration meetings are held within {n} weeks of the review "
            "cycle closing so the score scale is applied the same way everywhere.",
            "Anyone whose performance review score is the lowest on the scale twice "
            "in a row gets a {n}-week plan that is revisited at the next review.",
        ),
    ),
    (
        "equipment-requests",
        "Northwind Analytics Equipment Request Procedure (revision {n})",
        (
            "Staff request a laptop, monitor or headset from IT through the "
            "Northwind Analytics hardware portal, and IT answers each hardware "
            "request within {hours} hours.",
            "A standard laptop request from IT is approved automatically, and the "
            "laptop arrives from the IT stockroom within {n} weeks.",
            "A hardware request above a {amount} budget needs a written "
            "justification and sign-off from the budget owner before IT orders the "
            "equipment.",
            "IT refreshes every laptop every {n} years; a replacement laptop "
            "request can be filed {n} weeks before the refresh date.",
            "Damaged equipment is reported to IT within {hours} hours; IT ships a "
            "loaner laptop or monitor from stock while the hardware request is "
            "processed.",
        ),
    ),
)

# One question per topic slug; each reuses 4-6 of its topic's repeated words.
SYNTHETIC_QUESTIONS: dict[str, str] = {
    "vacation-policy": (
        "How many paid vacation days do employees accrue each year at "
        "Northwind Analytics?"
    ),
    "expense-reimbursement": (
        "Do I need a receipt for an expense claim under {amount} dollars to get "
        "reimbursement at Northwind Analytics?"
    ),
    "remote-work": (
        "How often can employees work remotely from home each week at "
        "Northwind Analytics?"
    ),
    "security-training": (
        "How often do staff finish the mandatory security module and phishing "
        "tests each quarter at Northwind Analytics?"
    ),
    "onboarding": (
        "What does the onboarding checklist cover for a hire and their buddy at "
        "Northwind Analytics?"
    ),
    "incident-response": (
        "Within how many minutes must the on-call engineers respond to a "
        "production incident alert at Northwind Analytics?"
    ),
    "data-retention": (
        "How long does Northwind Analytics retain customer data and backups?"
    ),
    "travel-booking": (
        "Do I require approval for a business trip flight before the airfare is "
        "booked at Northwind Analytics?"
    ),
    "performance-reviews": (
        "How often is a performance review held and how is the review score scale "
        "applied at Northwind Analytics?"
    ),
    "equipment-requests": (
        "Where do I request a laptop or monitor from IT at Northwind Analytics, "
        "and when does the hardware arrive?"
    ),
}

_REQUIRED_JSONL_KEYS = ("id", "title", "text")


def _slot_values(rng: random.Random) -> dict[str, int]:
    """Draw one value per numeric slot; called once per template to vary text."""
    return {
        "n": rng.randint(1, 30),
        "days": rng.randint(5, 30),
        "amount": 25 * rng.randint(1, 100),
        "hours": rng.choice((2, 4, 8, 24, 48, 72)),
        "pct": 5 * rng.randint(1, 19),
    }


def _synthetic_document(index: int, rng: random.Random) -> SourceDocument:
    slug, title_template, sentences = SYNTHETIC_TOPICS[index % len(SYNTHETIC_TOPICS)]
    title = title_template.format_map(_slot_values(rng))
    text = " ".join(s.format_map(_slot_values(rng)) for s in sentences)
    return SourceDocument(
        id=f"doc-{index:04d}",
        title=title,
        text=text,
        metadata={"topic": slug, "seq": index},
    )


def synthetic_corpus(n: int, seed: int = 1234) -> list[SourceDocument]:
    """Generate ``n`` deterministic handbook documents cycling through the topics.

    Example::

        docs = synthetic_corpus(20)          # ids doc-0000 .. doc-0019
        docs[0].metadata                     # {"topic": "vacation-policy", "seq": 0}
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    return [_synthetic_document(i, rng) for i in range(n)]


def synthetic_questions(n: int = 50, seed: int = 1234) -> list[str]:
    """Return ``n`` questions, one topic per question, cycling through the topics.

    Example::

        synthetic_questions(3)[0]
        # 'How many paid vacation days do employees accrue each year at ...'
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    slugs = [topic[0] for topic in SYNTHETIC_TOPICS]
    return [
        SYNTHETIC_QUESTIONS[slugs[i % len(slugs)]].format_map(_slot_values(rng))
        for i in range(n)
    ]


def _parse_jsonl_line(path: Path, lineno: int, line: str) -> SourceDocument:
    where = f"{path}: line {lineno}"
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: invalid JSON ({exc.msg}): {line[:80]!r}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected a JSON object, got {line[:80]!r}")
    missing = [key for key in _REQUIRED_JSONL_KEYS if key not in row]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}: {line[:80]!r}")
    try:
        return SourceDocument(
            id=row["id"],
            title=row["title"],
            text=row["text"],
            metadata=row.get("metadata") or {},
        )
    except ValidationError as exc:
        raise ValueError(f"{where}: {exc.errors()[0]['msg']}: {line[:80]!r}") from exc


def load_jsonl_corpus(path: Path) -> list[SourceDocument]:
    """Read one ``{"id","title","text","metadata"?}`` object per line.

    Blank lines are skipped; any other problem raises ``ValueError`` naming the
    line number and the first 80 characters of the offending line.

    Example::

        docs = load_jsonl_corpus(Path("corpus.jsonl"))
    """
    with Path(path).open(encoding="utf-8") as handle:
        return [
            _parse_jsonl_line(path, lineno, line)
            for lineno, line in enumerate(handle, start=1)
            if line.strip()
        ]


def _check_window(chunk_words: int, overlap_words: int) -> None:
    got = f"got chunk_words={chunk_words}, overlap_words={overlap_words}"
    if chunk_words < 1:
        raise ValueError(f"chunk_words must be >= 1; {got}")
    if overlap_words < 0:
        raise ValueError(f"overlap_words must be >= 0; {got}")
    if overlap_words >= chunk_words:
        raise ValueError(f"overlap_words must be smaller than chunk_words; {got}")


def _window_starts(n_words: int, chunk_words: int, overlap_words: int) -> list[int]:
    """Start offsets of each window; the final window may be shorter."""
    # A window starts at k*step only while the previous window stopped short of
    # the end (k*step < n - overlap); the first window always exists.
    step = chunk_words - overlap_words
    return list(range(0, max(n_words - overlap_words, 1), step))


def chunk_document(
    doc: SourceDocument, *, chunk_words: int = 120, overlap_words: int = 20
) -> list[Passage]:
    """Split a document into overlapping word windows.

    Example::

        chunk_document(doc, chunk_words=100, overlap_words=20)[1].id  # "doc-1:1"
    """
    _check_window(chunk_words, overlap_words)
    words = doc.text.split()
    if not words:
        return []
    starts = _window_starts(len(words), chunk_words, overlap_words)
    return [
        Passage(
            id=f"{doc.id}:{i}",
            text=" ".join(words[start : start + chunk_words]),
            metadata={**doc.metadata, "doc_id": doc.id, "title": doc.title, "chunk": i},
        )
        for i, start in enumerate(starts)
    ]
