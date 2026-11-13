"""Common CSV/XLSX exports for Replica and LIBERO, using audited API events.

XLSX uses only the standard library. Unknown and unrun measurements stay blank;
mock costs are separated from paid estimates. Output tokens already include
reasoning tokens; their optional subset cost must never be added a second time.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

EPISODE_FIELDS = ["id", "benchmark", "subset", "task_id", "variation", "repeat", "instruction",
    "model", "effort", "accounting_mode", "status", "termination_reasons", "steps", "simulated_seconds",
    "llm_calls", "http_attempts", "real_api_calls", "mock_calls", "usage_complete",
    "estimated_cost_usd", "known_cost_subtotal_usd", "mean_cost_per_call_usd",
    "reasoning_output_cost_usd", "simulated_cost_usd", "budget_charge_usd", "wall_seconds",
    "environment_success", "fidelity", "calibration_status", "initial_observation_sha256", "run_folder"]
CALL_FIELDS = ["run_id", "benchmark", "subset", "task_id", "variation", "repeat", "model", "effort",
    "accounting_mode", "call", "http_status", "latency_s", "input_tokens", "cached_input_tokens",
    "cache_write_tokens", "output_tokens", "reasoning_tokens", "estimated_cost_usd",
    "reasoning_output_cost_usd", "simulated_cost_usd", "budget_charge_usd", "usage_known"]
SUMMARY_FIELDS = ["benchmark", "model", "effort", "fidelity", "calibration_status", "accounting_mode",
    "episodes", "completed_episodes", "successful_episodes", "control_steps", "llm_calls",
    "http_attempts", "real_api_calls", "mock_calls", "unknown_usage_episodes",
    "known_cost_subtotal_usd", "total_cost_usd", "mean_cost_per_call_usd", "simulated_cost_usd"]


def _events(folder):
    path = folder/"api"/"api_usage.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def report_rows(root, data):
    root = Path(root)
    episodes, calls = [], []
    for run in data["runs"]:
        row = dict(run)
        events = _events(root/run["run_folder"])
        sent = {event["call"]: event for event in events if event["event"] == "request_started"}
        responses = {event["call"]: event for event in events if event["event"] == "response"}
        # A network failure may have no response; keep a per-call blank-cost row.
        call_ids = sorted(set(sent) | set(responses))
        row["llm_calls"] = len(call_ids)
        actual_attempts = sum(e["event"] == "http_attempt_started" for e in events)
        row["http_attempts"] = actual_attempts or len(call_ids) + sum(e["event"] == "server_error_retry" for e in events)
        mock = run.get("accounting_mode") == "mock"
        row["real_api_calls"], row["mock_calls"] = (0, len(call_ids)) if mock else (len(call_ids), 0)
        known = []
        reasoning_costs = []
        for call in call_ids:
            event = responses.get(call, {})
            usage = event.get("usage") or {}
            cost = event.get("estimated_usd")
            # BudgetTransport explicitly releases exhausted server-error responses.
            if cost is None and "released_usd" in event:
                cost = 0.
            valid_cost = cost is not None
            if valid_cost:
                known.append(cost)
            details = usage.get("output_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")
            price = data["protocol"].get("pricing_per_million", {}).get(run["model"])
            reasoning_cost = (reasoning_tokens * price[3] / 1e6
                              if isinstance(reasoning_tokens, int) and price else None)
            if reasoning_cost is not None:
                reasoning_costs.append(reasoning_cost)
            inherited = {k: row.get(k) for k in ("benchmark", "subset", "task_id", "variation", "repeat", "model", "effort", "accounting_mode")}
            call_row = dict(inherited, run_id=row["id"], call=call,
                http_status=event.get("http_status"), latency_s=event.get("latency_s"),
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens")
                    if run["model"].startswith("openai/") else usage.get("cache_read_input_tokens"),
                cache_write_tokens=(usage.get("input_tokens_details") or {}).get("cache_write_tokens")
                    if run["model"].startswith("openai/") else usage.get("cache_creation_input_tokens"),
                estimated_cost_usd=cost if not mock else None,
                reasoning_output_cost_usd=reasoning_cost if not mock else None,
                simulated_cost_usd=cost if mock else None,
                budget_charge_usd=event.get("budget_charge_usd"), usage_known=valid_cost)
            calls.append(call_row)
        attempted = run["status"] not in ("pending", "not_run", "not_run_budget")
        complete = attempted and len(known) == len(call_ids) and not run.get("usage_unknown", False)
        row.update(usage_complete=complete,
            estimated_cost_usd=sum(known) if complete and not mock else None,
            known_cost_subtotal_usd=sum(known) if attempted and not mock else None,
            mean_cost_per_call_usd=sum(known)/len(call_ids) if complete and call_ids and not mock else None,
            simulated_cost_usd=sum(known) if complete and mock else None,
            reasoning_output_cost_usd=sum(reasoning_costs) if complete and not mock and len(reasoning_costs) == len(call_ids) else None,
            simulated_seconds=row.get("steps", 0)/20 if row.get("steps") is not None else None)
        episodes.append(row)
    grouped = defaultdict(list)
    group_fields = SUMMARY_FIELDS[:6]
    for row in episodes:
        grouped[tuple(row.get(k) for k in group_fields)].append(row)
    summaries = []
    for group, rows in grouped.items():
        attempted = [r for r in rows if r["status"] not in ("pending", "not_run", "not_run_budget")]
        total_calls = sum(r["llm_calls"] for r in attempted)
        unknown = sum(not r["usage_complete"] for r in attempted)
        subtotal = sum(r["known_cost_subtotal_usd"] or 0 for r in attempted)
        paid = group[-1] == "paid"
        summaries.append(dict(zip(group_fields, group), episodes=len(rows),
            completed_episodes=sum(r["status"] in ("finished", "task_success") for r in rows),
            successful_episodes=sum(r.get("environment_success") is True for r in rows),
            control_steps=sum(r.get("steps") or 0 for r in attempted) if attempted else None,
            llm_calls=total_calls, http_attempts=sum(r["http_attempts"] for r in attempted),
            real_api_calls=sum(r["real_api_calls"] for r in attempted), mock_calls=sum(r["mock_calls"] for r in attempted),
            unknown_usage_episodes=unknown, known_cost_subtotal_usd=subtotal if paid and attempted else None,
            total_cost_usd=subtotal if paid and attempted and not unknown else None,
            mean_cost_per_call_usd=subtotal/total_calls if paid and total_calls and not unknown else None,
            simulated_cost_usd=sum(r["simulated_cost_usd"] or 0 for r in attempted) if not paid and attempted and not unknown else None))
    return episodes, calls, summaries


def write_reports(root):
    root = Path(root)
    data = json.loads((root/"evaluation.json").read_text())
    episodes, calls, summaries = report_rows(root, data)
    tables = [("Episodes", EPISODE_FIELDS, episodes), ("Calls", CALL_FIELDS, calls),
              ("By reasoning", SUMMARY_FIELDS, summaries)]
    for effort in ("low", "medium", "high"):
        tables.append((effort.capitalize(), EPISODE_FIELDS, [r for r in episodes if r["effort"] == effort]))
    protocol = data["protocol"]
    notes = {
        "agent_cost": "Sum of all input/cache/output API token costs; excludes local compute. Provider billing is authoritative.",
        "reasoning_output_cost": "Optional subset of output cost, already included in total; blank when provider does not report reasoning tokens.",
        "steps": "20 Hz simulator control ticks, including rejected waypoint ticks; excludes initial settling. Not API calls or task events.",
        "llm_calls": "Logical provider requests. http_attempts also counts transport retries. Mock calls are not real API calls.",
        "blank_cells": "Unknown usage, unrun episodes, and unavailable metrics stay blank. Subtotals exclude unknown costs.",
        "mock": "No provider connection or billing. Simulated token costs are shown only in simulated_cost_usd.",
        "fidelity": "Approximate geometry/layouts and proxy scores are provisional, not calibrated real-world benchmark results.",
        "grouping": "Summary groups benchmark, model, effort, fidelity, calibration status and accounting mode separately.",
        **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in protocol.items()}}
    tables.append(("Protocol", ["field", "value"], [dict(field=k, value=v) for k, v in notes.items()]))
    for title, fields, rows in tables[:3]:
        with (root/(title.lower().replace(" ", "_")+".csv")).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: _safe_csv(_value(v)) for k, v in row.items()} for row in rows)
    tmp = root/"evaluation.xlsx.tmp"
    _xlsx(tmp, tables)
    tmp.replace(root/"evaluation.xlsx")
    (root/"report_data.json").write_text(json.dumps(dict(protocol=protocol, episodes=episodes, calls=calls, summaries=summaries), indent=2))


def _value(value):
    return json.dumps(value) if isinstance(value, (list, dict)) else value


def _safe_csv(value):
    return "'"+value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value


def _column(index):
    result = ""
    while index:
        index, remainder = divmod(index-1, 26)
        result = chr(65+remainder)+result
    return result


def _xlsx(path, tables):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        overrides = ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(tables)+1))
        archive.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+overrides+'</Types>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{relns}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{relns}"><sheets>'+''.join(f'<sheet name="{escape(title)}" sheetId="{i}" r:id="rId{i}"/>' for i, (title, _, _) in enumerate(tables, 1))+'</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="{relns}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(tables)+1))+f'<Relationship Id="rId{len(tables)+1}" Type="{relns}/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', f'<styleSheet xmlns="{ns}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.00000000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="3"><xf fontId="0" fillId="0" borderId="0"/><xf fontId="1" fillId="2" borderId="0"/><xf fontId="0" fillId="0" borderId="0" numFmtId="164" applyNumberFormat="1"/></cellXfs></styleSheet>')
        for i, (_, fields, records) in enumerate(tables, 1):
            rows = [fields] + [[_value(row.get(key)) for key in fields] for row in records]
            contents = []
            for r, row in enumerate(rows, 1):
                cells = []
                for c, value in enumerate(row, 1):
                    ref = f'{_column(c)}{r}'
                    style = 1 if r == 1 else (2 if isinstance(value, float) else 0)
                    if value is None:
                        cells.append(f'<c r="{ref}"/>')
                    elif isinstance(value, bool):
                        cells.append(f'<c r="{ref}" t="b"><v>{int(value)}</v></c>')
                    elif isinstance(value, (int, float)):
                        cells.append(f'<c r="{ref}" s="{style}"><v>{value}</v></c>')
                    else:
                        clean = ''.join(ch for ch in str(value)[:32767] if ord(ch) >= 32 or ch in '\n\r\t')
                        cells.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(clean)}</t></is></c>')
                contents.append(f'<row r="{r}">'+''.join(cells)+'</row>')
            archive.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="{len(fields)}" width="22" customWidth="1"/></cols><sheetData>'+''.join(contents)+f'</sheetData><autoFilter ref="A1:{_column(len(fields))}{len(rows)}"/></worksheet>')
