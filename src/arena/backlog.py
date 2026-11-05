"""Dependency-free research backlog: validate, browse, update and render project memory."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "backlog" / "hypotheses.json"
DEFAULT_VIEW = REPO / "BACKLOG.md"
STATUSES = ("planned", "ready", "running", "done", "rejected", "deferred")
PRIORITIES = ("P0", "P1", "P2", "P3", "P4")
RISKS = ("low", "medium", "high")
RESOLVED = ("done", "rejected", "deferred")


class BacklogError(ValueError):
    """An invalid research record or unsafe workflow update."""


def _require(condition, message):
    if not condition:
        raise BacklogError(message)


def _range(value, name, nonnegative=False):
    _require(isinstance(value, list) and len(value) == 2, f"{name}: expected [low, high]")
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in value),
             f"{name}: expected finite numbers")
    _require(value[0] <= value[1], f"{name}: reversed range")
    _require(not nonnegative or value[0] >= 0, f"{name}: negative effort")


def validate(data):
    _require(isinstance(data, dict), "Backlog root must be an object")
    _require(data.get("schema_version") == 2, "Unsupported schema_version")
    _require(isinstance(data.get("items"), list), "Missing items list")
    for name in ("framing", "evaluation", "revisions", "references"):
        _require(isinstance(data.get(name), list) and data[name], f"Missing {name}")
        _require(all(isinstance(x, str) and x.strip() for x in data[name]), f"Invalid {name}")
    items = data["items"]
    _require(all(isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"]
                 for x in items), "Each item needs a nonempty string ID")
    ids = [x.get("id") for x in items]
    _require(len(ids) == len(set(ids)), "Duplicate item IDs")
    _require({f"H{i:02}" for i in range(1, 41)} <=
             {x["id"] for x in items if x.get("kind") == "hypothesis"},
             "Keep all forty stable hypothesis IDs H01–H40")
    ranks = [x.get("rank") for x in items]
    _require(all(type(x) is int and x > 0 for x in ranks) and len(set(ranks)) == len(ranks),
             "Ranks must be unique positive integers")
    by_id = {x["id"]: x for x in items}
    for item in items:
        key = item["id"]
        _require(item.get("kind") in ("hypothesis", "prerequisite"), f"{key}: invalid kind")
        prefix = "H" if item["kind"] == "hypothesis" else "E"
        _require(re.fullmatch(prefix + r"\d{2,}", key), f"{key}: invalid ID for item kind")
        for field, choices in (("status", STATUSES), ("priority", PRIORITIES), ("risk", RISKS)):
            _require(item.get(field) in choices, f"{key}: invalid {field}")
        for field in ("title", "claim", "basis", "first_test", "acceptance", "stop",
                      "risk_reason", "mitigation", "rationale", "revision"):
            _require(isinstance(item.get(field), str) and item[field].strip(), f"{key}: missing {field}")
        _require(item.get("estimate_confidence") in ("low", "medium"), f"{key}: invalid confidence")
        if item["kind"] == "hypothesis":
            _range(item.get("gain"), f"{key}.gain")
        else:
            _require(item.get("gain") is None, f"{key}: prerequisites have no score forecast")
        for field in ("engineering_hours", "pilot_gpu_wall_hours", "followup_gpu_wall_hours", "cpu_wall_hours"):
            _range(item.get(field), f"{key}.{field}", nonnegative=True)
        for field in ("depends_on", "evidence", "history"):
            _require(isinstance(item.get(field), list), f"{key}: missing {field} list")
        _require(all(isinstance(x, str) for x in item["depends_on"]), f"{key}: invalid dependency IDs")
        _require(all(isinstance(x, str) and x.strip() for x in item["evidence"]), f"{key}: invalid evidence")
        _require(all(x in by_id and x != key for x in item["depends_on"]), f"{key}: unknown/self dependency")
        _require(len(set(item["depends_on"])) == len(item["depends_on"]), f"{key}: duplicate dependency")
        _require(type(item.get("introduced_in")) is int and item["introduced_in"] > 0,
                 f"{key}: missing introduced_in version")
        _require(item.get("discovered_during") is None or
                 (type(item["discovered_during"]) is int and
                  0 < item["discovered_during"] < item["introduced_in"]),
                 f"{key}: invalid discovery version")
        _require(isinstance(item.get("derived_from"), list) and
                 all(isinstance(x, str) and x in by_id and x != key for x in item["derived_from"]),
                 f"{key}: invalid parent IDs")
        _require(all(by_id[x]["introduced_in"] < item["introduced_in"] for x in item["derived_from"]),
                 f"{key}: parent must come from an earlier version")
        for event in item["history"]:
            _require(isinstance(event, dict) and event.get("status") in STATUSES and
                     bool(event.get("at")) and bool(event.get("note")) and
                     isinstance(event.get("evidence"), list), f"{key}: malformed history event")
        if item["status"] in ("done", "rejected"):
            _require(any(e.get("status") == item["status"] and e.get("evidence") for e in item["history"]),
                     f"{key}: a closed item needs a result reference in its history")
    visiting, visited = set(), set()

    def visit(key):
        _require(key not in visiting, f"Dependency cycle at {key}")
        if key in visited:
            return
        visiting.add(key)
        for dep in by_id[key]["depends_on"]:
            visit(dep)
        visiting.remove(key)
        visited.add(key)

    for key in by_id:
        visit(key)
    for item in items:
        if item["status"] in ("ready", "running"):
            _require(not blockers(data, item), f"{item['id']}: ready/running item has unfinished dependencies")
    validate_versions(data)
    return data


def version_record(data, number=None):
    number = data["active_version"] if number is None else number
    record = next((x for x in data["versions"] if x["number"] == number), None)
    _require(record is not None, f"Unknown backlog version {number}")
    return record


def validate_versions(data):
    current = data.get("active_version")
    _require(type(current) is int and current > 0, "Invalid active_version")
    versions = data.get("versions")
    _require(isinstance(versions, list) and all(isinstance(x, dict) for x in versions),
             "Missing versions list")
    _require([x.get("number") for x in versions] == list(range(1, current + 2)),
             "Versions must be consecutive, with one next-version draft")
    by_id = {x["id"]: x for x in data["items"]}
    for record in versions:
        number = record["number"]
        _require(type(number) is int, "Version numbers must be integers")
        expected = "closed" if number < current else "active" if number == current else "draft"
        _require(record.get("state") == expected, f"v{number}: expected {expected} state")
        for field in ("item_ids", "focus_ids"):
            values = record.get(field)
            _require(isinstance(values, list) and all(isinstance(x, str) and x in by_id for x in values)
                     and len(set(values)) == len(values), f"v{number}: invalid {field}")
        _require(set(record["focus_ids"]) <= set(record["item_ids"]), f"v{number}: focus outside scope")
        _require(all(by_id[x]["introduced_in"] <= number for x in record["item_ids"]),
                 f"v{number}: includes future task")
        _require(isinstance(record.get("history"), list), f"v{number}: missing history")
        for stage in ("start", "final"):
            for suffix in ("sha256", "markdown_sha256"):
                digest = record.get(f"{stage}_{suffix}")
                _require(digest is None or (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)),
                         f"v{number}: invalid {stage} snapshot hash")
            _require(bool(record.get(f"{stage}_sha256")) == bool(record.get(f"{stage}_markdown_sha256")),
                     f"v{number}: snapshot needs both JSON and Markdown hashes")
        if number < current:
            _require(record.get("final_sha256") is not None, f"v{number}: closed span needs a snapshot")
    for item in data["items"]:
        introduced = item["introduced_in"]
        _require(introduced <= current + 1 and item["id"] in version_record(data, introduced)["item_ids"],
                 f"{item['id']}: missing from introduction version")
        if item["status"] in ("ready", "running"):
            _require(item["id"] in version_record(data)["item_ids"],
                     f"{item['id']}: future/historical task cannot be ready or running")
        if item["status"] == "done":
            _require(introduced <= current, f"{item['id']}: future task cannot be done")


def read(path=DEFAULT_DATA):
    with Path(path).open() as handle:
        return validate(json.load(handle))


def blockers(data, item):
    by_id = {x["id"]: x for x in data["items"]}
    return [key for key in item["depends_on"] if by_id[key]["status"] != "done"]


def select(data, *, statuses=(), priorities=(), risks=(), kind=None, max_engineering=None,
           min_gain=None, actionable=False, sort="priority", version=None, focus=False, cohort=None):
    record = version_record(data, version)
    scope = set(record["focus_ids"] if focus else record["item_ids"])
    items = [x for x in data["items"]
             if x["id"] in scope
             and (cohort != "new" or x["introduced_in"] == record["number"])
             and (cohort != "carried" or x["introduced_in"] < record["number"])
             and (not statuses or x["status"] in statuses)
             and (not priorities or x["priority"] in priorities)
             and (not risks or x["risk"] in risks)
             and (not kind or x["kind"] == kind)
             and (max_engineering is None or x["engineering_hours"][1] <= max_engineering)
             and (min_gain is None or (x["gain"] is not None and x["gain"][1] >= min_gain))
             and (not actionable or (record["state"] == "active" and not data.get("snapshot")
                                    and x["status"] in ("planned", "ready") and not blockers(data, x)))]
    if sort == "gain":
        key = lambda x: (-(x["gain"][1] if x["gain"] is not None else -math.inf), x["rank"])
    elif sort == "effort":
        key = lambda x: (x["engineering_hours"][1], x["rank"])
    else:
        key = lambda x: (PRIORITIES.index(x["priority"]), x["rank"])
    return sorted(items, key=key)


def span(numbers, signed=False):
    if numbers is None:
        return "—"
    fmt = lambda x: f"{x:+g}" if signed else f"{x:g}"
    return f"{fmt(numbers[0])}…{fmt(numbers[1])}"


def references(values):
    return "; ".join(x if x.startswith("[") else f"[{x}]({x})" for x in values)


def render(data, version=None, link_prefix=""):
    validate(data)
    record = version_record(data, version)
    number = record["number"]
    items = select(data, version=number)
    counts = Counter(x["status"] for x in items if x["kind"] == "hypothesis")
    state = f"{data['snapshot']['stage']} snapshot" if data.get("snapshot") else record["state"]
    lines = [f"# Research backlog v{number}", "", "<!-- Generated by scripts/backlog.py; edit canonical JSON or use the CLI. -->",
             "", f"Version **{number}**, {state}; **{len(items)}** tasks. Focus for this span: {', '.join(record['focus_ids']) or 'not selected'}.",
             "", "Original hypotheses H01–H40 retain their IDs. Follow-ups enter the next version; completed spans have preserved snapshots.",
             "", "Hypothesis status: " + ", ".join(f"{s} **{counts[s]}**" for s in STATUSES) + ".", ""]
    for paragraph in data["framing"]:
        lines += [paragraph, ""]
    if not data.get("snapshot") and number == data["active_version"]:
        lines += [f"Next version: **v{data['active_version'] + 1} draft**, in [NEXT.md](backlog/NEXT.md). "
                  "Use `versions` or `list --version N` to browse preserved spans.", ""]
    lines += ["## Browse and update", "",
        "Follow the [project policy](PROJECT_POLICY.md) for selecting work, preparing experiments, recording evidence and publishing results.",
        "", "```bash",
        "uv run python scripts/backlog.py list --priority P1",
        "uv run python scripts/backlog.py list --actionable",
        "uv run python scripts/backlog.py list --actionable --focus",
        "uv run python scripts/backlog.py versions",
        "uv run python scripts/backlog.py list --version next",
        "uv run python scripts/backlog.py list --max-engineering-hours 8 --sort effort",
        "uv run python scripts/backlog.py show H09",
        "# Start only an eligible task selected for the current span:",
        "# uv run python scripts/backlog.py update TASK_ID --status running --note 'Starting the predeclared pilot'",
        "# Close only after the work; --evidence points to the actual result/protocol:",
        "# uv run python scripts/backlog.py update TASK_ID --status done --note 'Actual outcome' --evidence docs/PATH_TO_ACTUAL_REPORT.md",
        "uv run python scripts/backlog.py check", "```", "",
        "The CLI uses the project's Python environment and standard library only. JSON is the source of truth; updates append a dated history and regenerate this page. Changes are reviewable with Git. No Kaggle job is launched by these commands.", "",
        "Each span has a fixed focus within its backlog version. Queue discoveries with `template` and `queue`; they are visible in the next-version draft and cannot run in this span. `advance` seals the current span, carries unfinished work forward and activates the next backlog. See the project policy and [current goal](GOAL.md) for the commands and completion rules.", "",
        "Statuses: `planned` → `ready` → `running` → `done`, with `deferred` and `rejected` available. Moving directly to `running` is allowed when dependencies are done. Closed items require a note and a result reference; `done` means the test finished, not that it improved the score. A dependency marked rejected does not unblock dependents. Do not change an estimate into a claimed result: put measurements, paired intervals and artifacts in the linked experiment report.", "",
        "`list` accepts repeated `--status`, `--priority`, and `--risk`, plus `--kind`, `--min-gain`, `--max-engineering-hours`, `--sort priority|gain|effort` and `--json`. Gain sorting uses the optimistic range endpoint, not expected return. `--actionable` hides deferred/closed/running items and unmet dependencies. Default priority includes the cost of learning whether an idea works; it is not just the largest forecast. `show ID --json` exposes the complete record. To change estimates or dependencies, edit the JSON, then run `render` and `check`.", "",
        "## Ranked work", "",
        "P0 = prerequisite; P1 = first pilot wave; P2 = follow-up candidates; P3 = conditional/low-cost research; P4 = defer until stronger evidence or resources. Ranks order work within a priority; they do not authorize running every item.", "",
        "Gain = planning range of **absolute log-loss reduction** (positive helps, negative regresses). Eng = person-hours; pilot/follow-up = elapsed hours of a **two-T4 session**, not summed device-hours. Follow-up cost is additional to the pilot. See each item's CPU estimate and qualifications below.", "",
        "| Rank | ID | Priority / status | Hypothesis | Gain | Risk | Eng h | Pilot GPU h | Follow-up GPU h | Needs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in items:
        lines.append(f"| {item['rank']} | [{item['id']}](#{item['id'].lower()}) | {item['priority']} / {item['status']} | {item['title']} | {span(item['gain'], True)} | {item['risk']} | {span(item['engineering_hours'])} | {span(item['pilot_gpu_wall_hours'])} | {span(item['followup_gpu_wall_hours'])} | {', '.join(item['depends_on']) or '—'} |")
    for title, key in (("Revisions to the original list", "revisions"), ("Evaluation and stopping rules", "evaluation")):
        lines += ["", f"## {title}", ""] + [f"- {x}" for x in data[key]]
    lines += ["", "## Work item details", ""]
    for item in items:
        lines += [f"### {item['id']}", "", f"**{item['title']}** — {item['priority']}, {item['status']}, risk {item['risk']}.", "",
            item["claim"], "", f"**Why this rank:** {item['rationale']}", "",
            f"**Evidence:** {item['basis']}", "",
            f"**Planning estimate:** loss reduction {span(item['gain'], True)}; confidence {item['estimate_confidence']}. Engineering {span(item['engineering_hours'])} person-hours; CPU {span(item['cpu_wall_hours'])} elapsed hours; two-T4 pilot {span(item['pilot_gpu_wall_hours'])} hours; additional follow-up {span(item['followup_gpu_wall_hours'])} hours.", "",
            f"**First test:** {item['first_test']}", "",
            f"**Advance/adopt:** {item['acceptance']}", "",
            f"**Stop/defer:** {item['stop']}", "",
            f"**Risk and mitigation:** {item['risk_reason']} {item['mitigation']}", "",
            f"**Revision:** {item['revision']}", "",
            f"**Depends on:** {', '.join(item['depends_on']) or 'none'}. Currently unmet: {', '.join(blockers(data, item)) or 'none'}.", "",
            f"**Origin:** introduced in v{item['introduced_in']}; discovered during {('v' + str(item['discovered_during'])) if item['discovered_during'] else 'initial planning'}; derived from {', '.join(item['derived_from']) or 'none'}.", "",
            f"**References:** {references(item['evidence'])}", ""]
        for event in item["history"]:
            lines += [f"- {event['at']} — {event['status']}: {event['note']}" +
                      (f" Evidence: {references(event['evidence'])}" if event['evidence'] else "")]
        if item["history"]:
            lines.append("")
    lines += ["## Sources and provenance", ""] + [f"- {x}" for x in data["references"]] + [""]
    output = "\n".join(lines)
    if link_prefix:
        output = re.sub(r"\]\((?!https?://|#|/)([^)]+)\)",
                        lambda match: f"]({link_prefix.rstrip('/')}/{match.group(1)})", output)
    return output


def atomic_write(path, text, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                _require(path.read_text() == text, f"Sealed snapshot cannot be overwritten: {path}")
        else:
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update(data, key, status, note, evidence=()):
    # Work on a copy so a rejected update never mutates the caller's record.
    candidate = json.loads(json.dumps(data))
    by_id = {x["id"]: x for x in candidate["items"]}
    _require(key in by_id, f"Unknown item {key}")
    ensure_live(data)
    _require(status in STATUSES, f"Invalid status {status}")
    _require(bool(note.strip()), "A note is required")
    _require(status not in ("done", "rejected") or bool(evidence),
             "Closing an item requires --evidence pointing to its result report")
    item = by_id[key]
    current = version_record(candidate)
    draft = version_record(candidate, candidate["active_version"] + 1)
    _require(key in current["item_ids"] or key in draft["item_ids"],
             f"{key}: historical task is closed; queue a follow-up instead")
    if status in ("ready", "running", "done"):
        _require(key in current["item_ids"], f"{key}: next-version task cannot execute in this span")
    if status == "done":
        _require(not blockers(candidate, item), f"{key}: cannot finish with unfinished dependencies")
    item["status"] = status
    item["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "status": status, "note": note.strip(), "evidence": list(evidence)})
    return validate(candidate)


def ensure_live(data):
    _require(not data.get("snapshot"), "Sealed snapshots are read-only; use the live backlog")


def dump(data):
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def snapshot_paths(source, number, stage):
    base = Path(source).parent / "versions" / f"v{number}.{stage}"
    return Path(str(base) + ".json"), Path(str(base) + ".md")


def seal_snapshot(data, source, stage, note, evidence=()):
    """Write-once pair; deterministic content allows retry after an interrupted save."""
    ensure_live(data)
    number = data["active_version"]
    json_path, markdown_path = snapshot_paths(source, number, stage)
    sealed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if json_path.exists():
        sealed_at = json.loads(json_path.read_text())["snapshot"]["sealed_at"]
    frozen = deepcopy(data)
    frozen["snapshot"] = {"version": number, "stage": stage, "note": note,
                          "evidence": list(evidence), "sealed_at": sealed_at}
    validate(frozen)
    content = dump(frozen)
    markdown = render(frozen, version=number, link_prefix="../..")
    atomic_write(json_path, content, immutable=True)
    atomic_write(markdown_path, markdown, immutable=True)
    return {f"{stage}_sha256": hashlib.sha256(content.encode()).hexdigest(),
            f"{stage}_markdown_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            f"{stage}_sealed_at": sealed_at}


def check_snapshots(data, source):
    for record in data["versions"]:
        for stage in ("start", "final"):
            digest = record.get(f"{stage}_sha256")
            if not digest:
                continue
            paths = snapshot_paths(source, record["number"], stage)
            for path, expected in zip(paths, (digest, record[f"{stage}_markdown_sha256"])):
                _require(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                         f"Snapshot hash mismatch or missing file: {path}")
            frozen = json.loads(paths[0].read_text())
            _require(frozen.get("snapshot", {}).get("stage") == stage and
                     frozen["snapshot"].get("version") == record["number"], "Wrong snapshot metadata")
            expected_ids = list(version_record(frozen, record["number"])["item_ids"])
            if stage == "start":
                # Explicit user planning can add work before a span starts. Keep
                # the original snapshot and account for every additive amendment.
                by_id = {x["id"]: x for x in data["items"]}
                for event in record["history"]:
                    if event.get("kind") != "planning_addition":
                        continue
                    additions = event.get("item_ids", [])
                    _require(additions and event.get("note") and event.get("evidence") and
                             event.get("at") and len(additions) == len(set(additions)) and
                             all(key not in expected_ids and key in by_id and
                                 by_id[key]["introduced_in"] == record["number"] for key in additions),
                             f"v{record['number']}: invalid planning addition")
                    expected_ids.extend(additions)
            _require(record["item_ids"] == expected_ids,
                     f"v{record['number']}: sealed version membership changed")


def load_version(data, source, number):
    record = version_record(data, number)
    if number < data["active_version"]:
        path, _ = snapshot_paths(source, number, "final")
        _require(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == record["final_sha256"],
                 f"Historical snapshot hash mismatch: {path}")
        frozen = read(path)
        _require(frozen.get("snapshot", {}).get("version") == number and
                 frozen["snapshot"].get("stage") == "final", "Invalid historical snapshot metadata")
        return frozen
    return data


def resolve_version(data, value):
    if value is None:
        return data["active_version"]
    if value == "next":
        return data["active_version"] + 1
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise BacklogError("Version must be a number or 'next'") from None
    version_record(data, number)
    return number


def task_template(kind="hypothesis"):
    return {"kind": kind, "title": "", "priority": "P2", "gain": None,
            "estimate_confidence": "low", "engineering_hours": None,
            "pilot_gpu_wall_hours": None, "followup_gpu_wall_hours": None, "cpu_wall_hours": None,
            "risk": "medium", "claim": "", "basis": "", "first_test": "", "acceptance": "",
            "stop": "", "risk_reason": "", "mitigation": "", "rationale": "", "revision": "",
            "depends_on": [], "evidence": []}


def queue(data, proposal, parents, note, evidence=(), *, into_active=False):
    ensure_live(data)
    _require(isinstance(proposal, dict), "Proposal must be a task object")
    _require(bool(note.strip()) and bool(parents), "Queueing needs a discovery note and parent task(s)")
    candidate = deepcopy(data)
    by_id = {x["id"]: x for x in candidate["items"]}
    _require(all(x in by_id and by_id[x]["introduced_in"] <= data["active_version"] for x in parents),
             "Follow-up parents must exist in the current or an earlier version")
    if into_active:
        record = version_record(candidate)
        _require(evidence, "Active planning additions need --evidence for the user-requested plan")
        _require(record.get("start_sealed_at") and not record["focus_ids"] and
                 not any(event.get("focus_ids") for event in record["history"]),
                 "Active additions require an unstarted span before any focus selection")
        _require(all(by_id[key]["status"] in ("planned", "deferred") for key in record["item_ids"]),
                 "Active additions require an unstarted span")
        started = datetime.fromisoformat(record["start_sealed_at"])
        _require(not any(event["status"] in ("ready", "running", "done", "rejected") and
                         datetime.fromisoformat(event["at"]) >= started
                         for key in record["item_ids"] for event in by_id[key]["history"]),
                 "Active additions require an unstarted span; previous work remains in history")
        _require(all(by_id[key]["introduced_in"] < data["active_version"] for key in parents),
                 "Active planning parents must come from an earlier version")
    kind = proposal.get("kind", "hypothesis")
    _require(kind in ("hypothesis", "prerequisite"), "Invalid proposed task kind")
    prefix = "H" if kind == "hypothesis" else "E"
    sequence = max([int(x[1:]) for x in by_id if x.startswith(prefix)], default=0) + 1
    key = f"{prefix}{sequence:02}"
    item = deepcopy(proposal)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    item.update(id=key, kind=kind, rank=max(x["rank"] for x in candidate["items"]) + 1,
                status="planned", introduced_in=data["active_version"] + (0 if into_active else 1),
                discovered_during=None if into_active else data["active_version"],
                derived_from=list(dict.fromkeys(parents)),
                history=[{"at": now,
                          "status": "planned", "note": note.strip(), "evidence": list(evidence)}])
    candidate["items"].append(item)
    version_record(candidate, item["introduced_in"])["item_ids"].append(key)
    if into_active:
        version_record(candidate)["history"].append({
            "at": now, "kind": "planning_addition", "item_ids": [key],
            "note": note.strip(), "evidence": list(evidence)})
    return validate(candidate), key


def set_focus(data, number, ids, note):
    ensure_live(data)
    _require(bool(note.strip()), "Changing focus needs a note")
    candidate = deepcopy(data)
    record = version_record(candidate, number)
    _require(record["state"] != "closed", "Historical focus is immutable")
    _require(ids and len(ids) == len(set(ids)) and set(ids) <= set(record["item_ids"]),
             "Focus must contain distinct tasks belonging to that version")
    _require(not any(x["status"] == "running" for x in data["items"]),
             "Finish or stop running work before changing span focus")
    record["focus_ids"] = list(ids)
    record["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "note": note.strip(), "focus_ids": list(ids)})
    return validate(candidate)


def advance(data, source, note, evidence):
    ensure_live(data)
    _require(note.strip() and evidence, "Closing a span needs a note and report reference")
    check_snapshots(data, source)
    current = version_record(data)
    _require(current["focus_ids"], "Select and resolve the span focus before advancing")
    _require(not any(x["status"] == "running" for x in data["items"]),
             "Cannot advance while a task is running")
    by_id = {x["id"]: x for x in data["items"]}
    for key in current["focus_ids"]:
        item = by_id[key]
        _require(item["status"] in RESOLVED, f"{key}: focus task has not been resolved")
        if item["status"] == "deferred":
            _require(any(x["status"] == "deferred" and x["evidence"] for x in item["history"]),
                     f"{key}: focus deferral needs a decision reference")
    candidate = deepcopy(data)
    record = version_record(candidate)
    # Seal before promotion; the archive contains the exact end-of-span registry.
    record.update(seal_snapshot(candidate, source, "final", note.strip(), evidence))
    record.update(state="closed", closed_at=record["final_sealed_at"],
                  closing_note=note.strip(), closing_evidence=list(evidence))
    next_record = version_record(candidate, candidate["active_version"] + 1)
    carry = [key for key in record["item_ids"] if by_id[key]["status"] not in ("done", "rejected")]
    next_record["item_ids"] = list(dict.fromkeys(carry + next_record["item_ids"]))
    next_record["state"] = "active"
    next_record["carried_from"] = record["number"]
    candidate["active_version"] += 1
    candidate["versions"].append({"number": candidate["active_version"] + 1, "state": "draft",
                                   "item_ids": [], "focus_ids": [], "history": []})
    validate(candidate)
    next_record.update(seal_snapshot(candidate, source, "start", "Start of next backlog span"))
    return validate(candidate)


def write_views(data, source, view):
    atomic_write(view, render(data))
    atomic_write(Path(source).parent / "NEXT.md",
                 render(data, version=data["active_version"] + 1, link_prefix=".."))


def save(data, source, view):
    ensure_live(data)
    validate(data)
    atomic_write(source, dump(data))
    write_views(data, source, view)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--view", type=Path, help="Markdown view; default BACKLOG.md for project data")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Browse, filter and rank work")
    listing.add_argument("--status", choices=STATUSES, action="append", default=[])
    listing.add_argument("--priority", choices=PRIORITIES, action="append", default=[])
    listing.add_argument("--risk", choices=RISKS, action="append", default=[])
    listing.add_argument("--kind", choices=("hypothesis", "prerequisite"))
    listing.add_argument("--max-engineering-hours", type=float)
    listing.add_argument("--min-gain", type=float, help="Minimum optimistic gain endpoint (not an expected gain)")
    listing.add_argument("--actionable", action="store_true")
    listing.add_argument("--sort", choices=("priority", "gain", "effort"), default="priority")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--version", help="Version number or 'next'; default active")
    listing.add_argument("--focus", action="store_true", help="Only this span's selected tasks")
    cohorts = listing.add_mutually_exclusive_group()
    cohorts.add_argument("--new", dest="cohort", action="store_const", const="new",
                         help="Only tasks introduced in this version")
    cohorts.add_argument("--carried", dest="cohort", action="store_const", const="carried",
                         help="Only unfinished tasks carried from earlier versions")
    showing = commands.add_parser("show")
    showing.add_argument("id")
    showing.add_argument("--json", action="store_true")
    showing.add_argument("--version", help="Read the state from a specified version")
    changing = commands.add_parser("update", help="Change status, append evidence/history and regenerate Markdown")
    changing.add_argument("id")
    changing.add_argument("--status", choices=STATUSES, required=True)
    changing.add_argument("--note", required=True)
    changing.add_argument("--evidence", action="append", default=[])
    commands.add_parser("versions", help="Show active, draft and archived spans")
    template = commands.add_parser("template", help="Print a blank follow-up task JSON object")
    template.add_argument("--kind", choices=("hypothesis", "prerequisite"), default="hypothesis")
    adding = commands.add_parser("queue", help="Record a discovery for the next version")
    adding.add_argument("--item-file", type=Path, required=True)
    adding.add_argument("--from", dest="parents", action="append", required=True)
    adding.add_argument("--note", required=True)
    adding.add_argument("--evidence", action="append", default=[])
    adding.add_argument("--into-active", action="store_true",
                        help="Explicit user planning addition to an unstarted active span; preserve its start snapshot")
    focusing = commands.add_parser("focus", help="Select the bounded scope of a span")
    focusing.add_argument("ids", nargs="+")
    focusing.add_argument("--version", help="Version number or 'next'; default active")
    focusing.add_argument("--note", required=True)
    advancing = commands.add_parser("advance", help="Seal this span and activate the next backlog")
    advancing.add_argument("--note", required=True)
    advancing.add_argument("--evidence", action="append", required=True)
    commands.add_parser("render", help="Regenerate Markdown from canonical JSON")
    commands.add_parser("check", help="Validate schema, dependencies and Markdown freshness")
    args = parser.parse_args(argv)
    view = args.view or (DEFAULT_VIEW if args.file.resolve() == DEFAULT_DATA.resolve()
                         else args.file.with_suffix(".md"))
    try:
        data = read(args.file)
        if args.command == "list":
            number = resolve_version(data, args.version)
            selected_data = load_version(data, args.file, number)
            items = select(selected_data, statuses=args.status, priorities=args.priority, risks=args.risk,
                           kind=args.kind, max_engineering=args.max_engineering_hours,
                           min_gain=args.min_gain, actionable=args.actionable, sort=args.sort,
                           version=number, focus=args.focus, cohort=args.cohort)
            if args.json:
                print(json.dumps(items, indent=2, ensure_ascii=False))
            else:
                print("ID   Pri Status    Risk   Gain             Eng h    Pilot GPU h  Title")
                for x in items:
                    print(f"{x['id']:<4} {x['priority']:<3} {x['status']:<9} {x['risk']:<6} "
                          f"{span(x['gain'], True):<16} {span(x['engineering_hours']):<8} "
                          f"{span(x['pilot_gpu_wall_hours']):<12} {x['title']}")
                print(f"{len(items)} items; gain ranges are unmeasured planning estimates.")
        elif args.command == "show":
            selected_data = load_version(data, args.file, resolve_version(data, args.version))
            item = next((x for x in selected_data["items"] if x["id"] == args.id), None)
            _require(item is not None, f"Unknown item {args.id}")
            if args.version is not None:
                _require(item["id"] in version_record(selected_data, resolve_version(data, args.version))["item_ids"],
                         f"{args.id}: not part of that version")
            if args.json:
                print(json.dumps(item, indent=2, ensure_ascii=False))
            else:
                print(f"{item['id']}: {item['title']} ({item['priority']}, {item['status']})")
                for field in ("claim", "gain", "estimate_confidence", "risk", "risk_reason", "mitigation",
                              "engineering_hours", "pilot_gpu_wall_hours", "followup_gpu_wall_hours",
                              "cpu_wall_hours", "rationale", "basis", "first_test", "acceptance", "stop",
                              "revision", "depends_on", "introduced_in", "discovered_during", "derived_from",
                              "evidence", "history"):
                    print(f"{field}: {item[field]}")
                print(f"unmet_dependencies: {blockers(selected_data, item)}")
        elif args.command == "check":
            check_snapshots(data, args.file)
            _require(view.exists() and view.read_text() == render(data),
                     f"Markdown is stale or missing: {view}; run render")
            next_view = args.file.parent / "NEXT.md"
            _require(next_view.exists() and next_view.read_text() ==
                     render(data, version=data["active_version"] + 1, link_prefix=".."),
                     f"Next-version Markdown is stale or missing: {next_view}; run render")
            print(f"OK: v{data['active_version']}, {len(data['items'])} registered tasks, "
                  "acyclic dependencies, verified snapshots, current Markdown")
        elif args.command == "render":
            ensure_live(data)
            write_views(data, args.file, view)
            print(f"Rendered {view}")
        elif args.command == "versions":
            print("Version  State   Tasks  Focus  New discoveries")
            for record in data["versions"]:
                discoveries = sum(x["introduced_in"] == record["number"] and
                                  x["discovered_during"] is not None for x in data["items"])
                print(f"v{record['number']:<7} {record['state']:<7} {len(record['item_ids']):<6} "
                      f"{len(record['focus_ids']):<6} {discoveries}")
        elif args.command == "template":
            print(dump(task_template(args.kind)), end="")
        elif args.command == "update":
            check_snapshots(data, args.file)
            candidate = update(data, args.id, args.status, args.note, args.evidence)
            save(candidate, args.file, view)
            print(f"Updated {args.id} → {args.status}; regenerated {view}")
        elif args.command == "queue":
            check_snapshots(data, args.file)
            proposal = json.loads(args.item_file.read_text())
            candidate, key = queue(data, proposal, args.parents, args.note, args.evidence,
                                   into_active=args.into_active)
            save(candidate, args.file, view)
            if args.into_active:
                print(f"Added {key} to unstarted v{data['active_version']}; original start snapshot preserved")
            else:
                print(f"Queued {key} for v{data['active_version'] + 1}, discovered during v{data['active_version']}")
        elif args.command == "focus":
            check_snapshots(data, args.file)
            candidate = set_focus(data, resolve_version(data, args.version), args.ids, args.note)
            save(candidate, args.file, view)
            print(f"Updated span focus; regenerated {view}")
        else:
            candidate = advance(data, args.file, args.note, args.evidence)
            save(candidate, args.file, view)
            print(f"Sealed v{data['active_version']}; active backlog is now v{candidate['active_version']}")
    except (BacklogError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"backlog: {exc}\n")


if __name__ == "__main__":
    main()
