import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from backtestlint import cli
from backtestlint.linter import collect_files, lint_file, notebook_source
from backtestlint.rules import RULES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD = "import pandas as pd\ny = df.close.shift(-1)\nz = df.close.rolling(5, center=True).mean()\nx = s.bfill()\n"


def notebook(*cells):
    return json.dumps({"cells": [{"cell_type": t, "source": src} for t, src in cells], "nbformat": 4, "nbformat_minor": 5, "metadata": {}})


class CliBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def run_cli(self, *argv):
        out = io.StringIO()
        with mock.patch("sys.stderr", io.StringIO()) as err:
            code = cli.main(list(argv), out=out)
        return code, out.getvalue(), err.getvalue()


class CheckCommandTests(CliBase):
    def test_findings_text_and_exit_code(self):
        p = self.write("bt.py", BAD)
        code, out, _ = self.run_cli("check", p, "--color", "never")
        self.assertEqual(code, 1)
        for part in ("BL001", "BL002", "BL008", "bt.py:2:5", "3 finding(s)", "high confidence"):
            self.assertIn(part, out)

    def test_default_command_and_clean_file(self):
        p = self.write("ok.py", "x = df.close.shift(1)\n")
        self.assertEqual(self.run_cli(p)[0], 0)
        self.assertIn("0 finding(s)", self.run_cli(p)[1])
        self.assertEqual(self.run_cli(self.write("bad.py", BAD), "--exit-zero")[0], 0)

    def test_directories_and_skipped_folders(self):
        self.write("pkg/a.py", "y = s.shift(-1)\n")
        self.write("pkg/sub/b.py", "y = s.bfill()\n")
        self.write("pkg/.venv/lib/c.py", "y = s.shift(-1)\n")
        self.write("pkg/node_modules/d.py", "y = s.shift(-1)\n")
        self.write("pkg/readme.txt", "y = s.shift(-1)\n")
        files = [os.path.relpath(f, self.dir) for f in collect_files([os.path.join(self.dir, "pkg")])]
        self.assertEqual(sorted(files), ["pkg/a.py", "pkg/sub/b.py"])
        code, out, _ = self.run_cli(os.path.join(self.dir, "pkg"), "--format", "json")
        self.assertEqual(len(json.loads(out)), 2)

    def test_filters(self):
        p = self.write("bt.py", BAD + "sig = a\nr = sig * ret\n")
        rules = lambda *a: {f["rule"] for f in json.loads(self.run_cli(p, "--format", "json", *a)[1])}
        self.assertEqual(rules("--select", "BL001"), {"BL001"})
        self.assertNotIn("BL001", rules("--ignore", "BL001"))
        self.assertNotIn("BL004", rules("--min-severity", "HIGH"))
        self.assertNotIn("BL004", rules("--min-confidence", "MEDIUM"))
        self.assertEqual(rules("--select", "BL001", "--ignore", "BL001"), set())

    def test_json_output_shape(self):
        data = json.loads(self.run_cli(self.write("bt.py", BAD), "--format", "json")[1])
        first = data[0]
        for key in ("rule", "name", "severity", "confidence", "path", "line", "column", "message", "fix"):
            self.assertIn(key, first)
        self.assertEqual((first["rule"], first["line"]), ("BL001", 2))

    def test_sarif_output(self):
        data = json.loads(self.run_cli(self.write("bt.py", BAD), "--format", "sarif")[1])
        self.assertEqual(data["version"], "2.1.0")
        run = data["runs"][0]
        self.assertEqual(run["tool"]["driver"]["name"], "backtestlint")
        self.assertEqual(len(run["tool"]["driver"]["rules"]), len(RULES))
        r = run["results"][0]
        self.assertEqual((r["ruleId"], r["level"]), ("BL001", "error"))
        loc = r["locations"][0]["physicalLocation"]
        self.assertEqual((loc["region"]["startLine"], loc["region"]["startColumn"]), (2, 5))

    def test_color_flag(self):
        p = self.write("bt.py", BAD)
        self.assertIn("\033[", self.run_cli(p, "--color", "always")[1])
        self.assertNotIn("\033[", self.run_cli(p, "--color", "never")[1])

    def test_unreadable_and_broken_files(self):
        broken = self.write("broken.py", "def f(:\n")
        code, out, err = self.run_cli(broken)
        self.assertEqual(code, 2)
        self.assertIn("cannot analyse", err)
        good = self.write("good.py", "y = s.shift(-1)\n")
        code, out, err = self.run_cli(broken, good)
        self.assertEqual(code, 1)                                  # findings win over the broken file
        self.assertIn("1 file(s) could not be analysed", out)
        self.assertEqual(self.run_cli(os.path.join(self.dir, "missing.py"))[0], 2)

    def test_unknown_rules_are_errors(self):
        p = self.write("a.py", "x = 1\n")
        for flag in ("--select", "--ignore"):
            code, _, err = self.run_cli(p, flag, "BL999")
            self.assertEqual(code, 2)
            self.assertIn("unknown rule", err)


class NotebookTests(CliBase):
    def test_code_cells_only_and_magics_stripped(self):
        text = notebook(("markdown", ["# y = s.shift(-1)"]), ("code", ["%matplotlib inline\n", "import pandas as pd\n", "!pip list"]),
                        ("code", "a = 1\ny = s.shift(-1)\n"))
        source, cellmap = notebook_source(text)
        self.assertNotIn("%matplotlib", source)
        self.assertNotIn("pip list", source)
        p = self.write("nb.ipynb", text)
        found = lint_file(p)
        self.assertEqual([f.rule_id for f in found], ["BL001"])
        self.assertIn("(cell 3)", found[0].message)               # cell numbering counts markdown cells too

    def test_notebook_via_cli_and_bad_json(self):
        p = self.write("nb.ipynb", notebook(("code", "y = s.shift(-1)\n")))
        self.assertEqual(self.run_cli(p)[0], 1)
        bad = self.write("bad.ipynb", "{not json")
        code, _, err = self.run_cli(bad)
        self.assertEqual(code, 2)

    def test_empty_notebook(self):
        self.assertEqual(self.run_cli(self.write("e.ipynb", notebook()))[0], 0)


class InfoCommandTests(CliBase):
    def test_rules_and_explain(self):
        code, out, _ = self.run_cli("rules")
        self.assertEqual(len(out.splitlines()), len(RULES))
        code, out, _ = self.run_cli("explain", "bl001")
        for part in ("negative-shift", "Why it matters", "How to fix"):
            self.assertIn(part, out)
        self.assertEqual(self.run_cli("explain", "BL999")[0], 2)

    def test_every_rule_is_documented(self):
        for r in RULES.values():
            self.assertTrue(len(r.summary) > 15 and len(r.why) > 20 and len(r.fix) > 15, r.id)
            self.assertIn(r.severity, ("HIGH", "MEDIUM", "LOW"))

    def test_no_arguments_prints_help(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn("usage", out.lower())

    def test_module_entry_point(self):
        r = subprocess.run([sys.executable, "-m", "backtestlint", "--version"], cwd=ROOT, capture_output=True, text=True)
        self.assertIn("1.0.0", r.stdout + r.stderr)


class SelfLintTests(unittest.TestCase):
    def test_the_linter_passes_its_own_rules(self):
        files = list(collect_files([os.path.join(ROOT, "backtestlint")]))
        self.assertTrue(files)
        for f in files:
            self.assertEqual([x.rule_id for x in lint_file(f)], [], f)


if __name__ == "__main__":
    unittest.main()
