"""Command line interface."""
import argparse
import json
import sys

from . import __version__
from .linter import collect_files, lint_file
from .rules import CONFIDENCES, RULES, SEVERITIES

ANSI = {"HIGH": "\033[1;31m", "MEDIUM": "\033[33m", "LOW": "\033[36m", "end": "\033[0m"}


def build_parser():
    ap = argparse.ArgumentParser(prog="backtestlint", description="Find look-ahead bias and data leakage in Python backtest code.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("check", help="lint files, folders and notebooks (default command)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    p.add_argument("--min-severity", choices=SEVERITIES, default="LOW")
    p.add_argument("--min-confidence", choices=CONFIDENCES, default="LOW")
    p.add_argument("--select", action="append", default=[], metavar="RULE", help="only these rules (repeatable)")
    p.add_argument("--ignore", action="append", default=[], metavar="RULE", help="skip these rules (repeatable)")
    p.add_argument("--exit-zero", action="store_true", help="always exit 0 (report only)")
    p.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    sub.add_parser("rules", help="list all rules")
    p = sub.add_parser("explain", help="explain one rule")
    p.add_argument("rule")
    return ap


def sarif(findings):
    rules = [{"id": r.id, "name": r.name, "shortDescription": {"text": r.summary}, "fullDescription": {"text": r.why},
              "help": {"text": r.fix}} for r in RULES.values()]
    level = {"HIGH": "error", "MEDIUM": "warning", "LOW": "note"}
    results = [{"ruleId": f.rule_id, "level": level[f.severity], "message": {"text": f.message},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": f.path.replace("\\", "/")},
                                                    "region": {"startLine": f.line, "startColumn": f.col + 1}}}]} for f in findings]
    return json.dumps({"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
                       "runs": [{"tool": {"driver": {"name": "backtestlint", "version": __version__, "rules": rules}}, "results": results}]}, indent=2)


def run_check(args, out):
    for r in args.select + args.ignore:
        if r not in RULES:
            raise KeyError("unknown rule: %s" % r)
    findings, errors = [], 0
    sev_ok, conf_ok = SEVERITIES.index(args.min_severity), CONFIDENCES.index(args.min_confidence)
    for path in collect_files(args.paths):
        try:
            found = lint_file(path)
        except (SyntaxError, OSError, ValueError) as exc:
            sys.stderr.write("backtestlint: cannot analyse %s\n" % exc if isinstance(exc, SyntaxError) else "backtestlint: %s: %s\n" % (path, exc))
            errors += 1
            continue
        findings += [f for f in found if (not args.select or f.rule_id in args.select) and f.rule_id not in args.ignore
                     and SEVERITIES.index(f.severity) <= sev_ok and CONFIDENCES.index(f.confidence) <= conf_ok]
    color = args.color == "always" or (args.color == "auto" and out.isatty())
    if args.format == "json":
        out.write(json.dumps([f.to_dict() for f in findings], indent=2) + "\n")
    elif args.format == "sarif":
        out.write(sarif(findings) + "\n")
    else:
        for f in findings:
            sev = "%s%-6s%s" % (ANSI[f.severity], f.severity, ANSI["end"]) if color else "%-6s" % f.severity
            out.write("%s:%d:%d: %s %s [%s confidence] %s\n" % (f.path, f.line, f.col + 1, f.rule_id, sev, f.confidence.lower(), f.message))
        out.write("backtestlint: %d finding(s)%s\n" % (len(findings), "; %d file(s) could not be analysed" % errors if errors else ""))
    if errors and not findings:
        return 2
    return 0 if (args.exit_zero or not findings) else 1


def run(argv=None, out=None):
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("check", "rules", "explain", "-h", "--help", "--version"):
        argv.insert(0, "check")
    args = build_parser().parse_args(argv)
    if args.cmd == "rules":
        for r in RULES.values():
            out.write("%s  %-6s  %-22s %s\n" % (r.id, r.severity, r.name, r.summary))
        return 0
    if args.cmd == "explain":
        r = RULES.get(args.rule.upper())
        if not r:
            raise KeyError("unknown rule: %s" % args.rule)
        out.write("%s  %s  (%s)\n\n%s\n\nWhy it matters:\n  %s\n\nHow to fix:\n  %s\n" % (r.id, r.summary, r.severity, r.name, r.why, r.fix))
        return 0
    if args.cmd is None:
        build_parser().print_help(out)
        return 2
    return run_check(args, out)


def main(argv=None, out=None):
    try:
        return run(argv, out)
    except KeyError as exc:
        sys.stderr.write("backtestlint: error: %s\n" % exc.args[0])
        return 2
