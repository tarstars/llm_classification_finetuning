"""Dependency-free research backlog: validate, browse, update and render project memory."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "backlog" / "hypotheses.json"
DEFAULT_VIEW = REPO / "BACKLOG.md"
STATUSES = ("planned", "ready", "running", "done", "rejected", "deferred")
PRIORITIES = ("P0", "P1", "P2", "P3", "P4")
RISKS = ("low", "medium", "high")


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
    _require(data.get("schema_version") == 1, "Unsupported schema_version")
    _require(isinstance(data.get("items"), list), "Missing items list")
    for name in ("framing", "evaluation", "revisions", "references"):
        _require(isinstance(data.get(name), list) and data[name], f"Missing {name}")
        _require(all(isinstance(x, str) and x.strip() for x in data[name]), f"Invalid {name}")
    items = data["items"]
    _require(all(isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"]
                 for x in items), "Each item needs a nonempty string ID")
    ids = [x.get("id") for x in items]
    _require(len(ids) == len(set(ids)), "Duplicate item IDs")
    _require({f"H{i:02}" for i in range(1, 41)} ==
             {x["id"] for x in items if x.get("kind") == "hypothesis"},
             "Keep all forty stable hypothesis IDs H01–H40")
    ranks = [x.get("rank") for x in items]
    _require(all(type(x) is int and x > 0 for x in ranks) and len(set(ranks)) == len(ranks),
             "Ranks must be unique positive integers")
    by_id = {x["id"]: x for x in items}
    for item in items:
        key = item["id"]
        _require(item.get("kind") in ("hypothesis", "prerequisite"), f"{key}: invalid kind")
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
    return data


def read(path=DEFAULT_DATA):
    with Path(path).open() as handle:
        return validate(json.load(handle))


def blockers(data, item):
    by_id = {x["id"]: x for x in data["items"]}
    return [key for key in item["depends_on"] if by_id[key]["status"] != "done"]


def select(data, *, statuses=(), priorities=(), risks=(), kind=None, max_engineering=None,
           min_gain=None, actionable=False, sort="priority"):
    items = [x for x in data["items"]
             if (not statuses or x["status"] in statuses)
             and (not priorities or x["priority"] in priorities)
             and (not risks or x["risk"] in risks)
             and (not kind or x["kind"] == kind)
             and (max_engineering is None or x["engineering_hours"][1] <= max_engineering)
             and (min_gain is None or (x["gain"] is not None and x["gain"][1] >= min_gain))
             and (not actionable or (x["status"] in ("planned", "ready") and not blockers(data, x)))]
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


def render(data):
    validate(data)
    counts = Counter(x["status"] for x in data["items"] if x["kind"] == "hypothesis")
    lines = ["# Research backlog", "", "<!-- Generated by scripts/backlog.py render; edit backlog/hypotheses.json or use the CLI. -->",
             "", "Forty stable hypotheses plus one evaluation prerequisite. No new model experiment was run to create this backlog.",
             "", "Hypothesis status: " + ", ".join(f"{s} **{counts[s]}**" for s in STATUSES) + ".", ""]
    for paragraph in data["framing"]:
        lines += [paragraph, ""]
    lines += ["## Browse and update", "", "```bash",
        "uv run python scripts/backlog.py list --priority P1",
        "uv run python scripts/backlog.py list --actionable",
        "uv run python scripts/backlog.py list --max-engineering-hours 8 --sort effort",
        "uv run python scripts/backlog.py show H09",
        "uv run python scripts/backlog.py update E00 --status running --note 'Preparing the matched evaluation protocol'",
        "# Close only after the work; --evidence points to the actual result/protocol:",
        "# uv run python scripts/backlog.py update E00 --status done --note 'Protocol frozen' --evidence docs/PATH_TO_ACTUAL_PROTOCOL.md",
        "uv run python scripts/backlog.py check", "```", "",
        "The CLI uses the project's Python environment and standard library only. JSON is the source of truth; updates append a dated history and regenerate this page. Changes are reviewable with Git. No Kaggle job is launched by these commands.", "",
        "Statuses: `planned` → `ready` → `running` → `done`, with `deferred` and `rejected` available. Moving directly to `running` is allowed when dependencies are done. Closed items require a note and a result reference; `done` means the test finished, not that it improved the score. A dependency marked rejected does not unblock dependents. Do not change an estimate into a claimed result: put measurements, paired intervals and artifacts in the linked experiment report.", "",
        "`list` accepts repeated `--status`, `--priority`, and `--risk`, plus `--kind`, `--min-gain`, `--max-engineering-hours`, `--sort priority|gain|effort` and `--json`. Gain sorting uses the optimistic range endpoint, not expected return. `--actionable` hides deferred/closed/running items and unmet dependencies. Default priority includes the cost of learning whether an idea works; it is not just the largest forecast. `show ID --json` exposes the complete record. To change estimates or dependencies, edit the JSON, then run `render` and `check`.", "",
        "## Ranked work", "",
        "P0 = prerequisite; P1 = first pilot wave; P2 = follow-up candidates; P3 = conditional/low-cost research; P4 = defer until stronger evidence or resources. Ranks order work within a priority; they do not authorize running every item.", "",
        "Gain = planning range of **absolute log-loss reduction** (positive helps, negative regresses). Eng = person-hours; pilot/follow-up = elapsed hours of a **two-T4 session**, not summed device-hours. Follow-up cost is additional to the pilot. See each item's CPU estimate and qualifications below.", "",
        "| Rank | ID | Priority / status | Hypothesis | Gain | Risk | Eng h | Pilot GPU h | Follow-up GPU h | Needs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in select(data):
        lines.append(f"| {item['rank']} | [{item['id']}](#{item['id'].lower()}) | {item['priority']} / {item['status']} | {item['title']} | {span(item['gain'], True)} | {item['risk']} | {span(item['engineering_hours'])} | {span(item['pilot_gpu_wall_hours'])} | {span(item['followup_gpu_wall_hours'])} | {', '.join(item['depends_on']) or '—'} |")
    for title, key in (("Revisions to the original list", "revisions"), ("Evaluation and stopping rules", "evaluation")):
        lines += ["", f"## {title}", ""] + [f"- {x}" for x in data[key]]
    lines += ["", "## Work item details", ""]
    for item in select(data):
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
            f"**References:** {references(item['evidence'])}", ""]
        for event in item["history"]:
            lines += [f"- {event['at']} — {event['status']}: {event['note']}" +
                      (f" Evidence: {references(event['evidence'])}" if event['evidence'] else "")]
        if item["history"]:
            lines.append("")
    lines += ["## Sources and provenance", ""] + [f"- {x}" for x in data["references"]] + [""]
    return "\n".join(lines)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update(data, key, status, note, evidence=()):
    # Work on a copy so a rejected update never mutates the caller's record.
    candidate = json.loads(json.dumps(data))
    by_id = {x["id"]: x for x in candidate["items"]}
    _require(key in by_id, f"Unknown item {key}")
    _require(status in STATUSES, f"Invalid status {status}")
    _require(bool(note.strip()), "A note is required")
    _require(status not in ("done", "rejected") or bool(evidence),
             "Closing an item requires --evidence pointing to its result report")
    item = by_id[key]
    if status == "done":
        _require(not blockers(candidate, item), f"{key}: cannot finish with unfinished dependencies")
    item["status"] = status
    item["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "status": status, "note": note.strip(), "evidence": list(evidence)})
    return validate(candidate)


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
    showing = commands.add_parser("show")
    showing.add_argument("id")
    showing.add_argument("--json", action="store_true")
    changing = commands.add_parser("update", help="Change status, append evidence/history and regenerate Markdown")
    changing.add_argument("id")
    changing.add_argument("--status", choices=STATUSES, required=True)
    changing.add_argument("--note", required=True)
    changing.add_argument("--evidence", action="append", default=[])
    commands.add_parser("render", help="Regenerate Markdown from canonical JSON")
    commands.add_parser("check", help="Validate schema, dependencies and Markdown freshness")
    args = parser.parse_args(argv)
    view = args.view or (DEFAULT_VIEW if args.file.resolve() == DEFAULT_DATA.resolve()
                         else args.file.with_suffix(".md"))
    try:
        data = read(args.file)
        if args.command == "list":
            items = select(data, statuses=args.status, priorities=args.priority, risks=args.risk,
                           kind=args.kind, max_engineering=args.max_engineering_hours,
                           min_gain=args.min_gain, actionable=args.actionable, sort=args.sort)
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
            item = next((x for x in data["items"] if x["id"] == args.id), None)
            _require(item is not None, f"Unknown item {args.id}")
            if args.json:
                print(json.dumps(item, indent=2, ensure_ascii=False))
            else:
                print(f"{item['id']}: {item['title']} ({item['priority']}, {item['status']})")
                for field in ("claim", "gain", "estimate_confidence", "risk", "risk_reason", "mitigation",
                              "engineering_hours", "pilot_gpu_wall_hours", "followup_gpu_wall_hours",
                              "cpu_wall_hours", "rationale", "basis", "first_test", "acceptance", "stop",
                              "revision", "depends_on", "evidence", "history"):
                    print(f"{field}: {item[field]}")
                print(f"unmet_dependencies: {blockers(data, item)}")
        elif args.command == "check":
            _require(view.exists() and view.read_text() == render(data),
                     f"Markdown is stale or missing: {view}; run render")
            print(f"OK: {len(data['items'])} items, stable H01–H40, acyclic dependencies, current Markdown")
        elif args.command == "render":
            atomic_write(view, render(data))
            print(f"Rendered {view}")
        else:
            candidate = update(data, args.id, args.status, args.note, args.evidence)
            markdown = render(candidate)
            atomic_write(args.file, json.dumps(candidate, indent=2, ensure_ascii=False) + "\n")
            atomic_write(view, markdown)
            print(f"Updated {args.id} → {args.status}; regenerated {view}")
    except (BacklogError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"backlog: {exc}\n")


if __name__ == "__main__":
    main()
