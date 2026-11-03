"""Read native legacy trace_results.json, preserving the fixed case denominator."""
from pathlib import Path
import json
from trajectory_retention.contracts import CheckResult, CheckStatus

def observe(output: Path) -> dict:
    path = Path(output) / "verifier/trace_results.json"
    try:
        doc = json.loads(path.read_text()) if path.is_file() else None
        return {"native":doc, "collect_errors":[], "parse_error":None}
    except (ValueError, OSError) as exc:
        return {"native":None, "collect_errors":[], "parse_error":str(exc)}

def translate(expected, observation, receipt):
    doc = observation['native']
    error = receipt.get('infrastructure_error') or observation.get('parse_error')
    if doc is not None and not isinstance(doc, dict): error = 'Invalid native trace document'
    cases = (doc or {}).get('cases', {}) if not error else {}
    if not error and doc:
        if doc.get('total_cases') != len(expected): error = 'Native case denominator differs from frozen cases'
        if not isinstance(cases, dict): error = 'Invalid native cases'
        elif any(not isinstance(v, dict) or type(v.get('passed')) is not bool for v in cases.values()):
            error = 'Invalid native case result'
        elif not doc.get('fatal_error') and set(expected) - set(cases): error = 'Native case results incomplete'
        elif not doc.get('fatal_error') and doc.get('passed_cases') != sum(cases[c]['passed'] for c in expected):
            error = 'Native per-case totals inconsistent'
    if error:
        return tuple(CheckResult(check_id=c,status=CheckStatus.INFRA_ERROR,details={'reachable':None,'reason':error}) for c in expected)
    result=[]
    for c in expected:
        row = cases.get(c)
        if row is None:
            result.append(CheckResult(check_id=c,status=CheckStatus.BLOCKED,raw_status='native_grader_before_cases',
                details={'reachable':False,'blocking_reason':(doc or {}).get('fatal_error','native grader did not report cases')}))
        else:
            result.append(CheckResult(check_id=c,status=CheckStatus.PASS if row['passed'] else CheckStatus.FAIL,
                raw_status='passed' if row['passed'] else 'failed',
                details={'reachable':True,'diagnostic_tail':str(row.get('errors',[]))[-600:]}))
    return tuple(result)
