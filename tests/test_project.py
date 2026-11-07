import os
import re
import unittest

import backtestlint
from backtestlint.rules import RULES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()

class ProjectConsistencyTests(unittest.TestCase):
    def test_version_matches_pyproject_and_changelog(self):
        version = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
        self.assertEqual(version, backtestlint.__version__)
        self.assertIn("## " + version, read("CHANGELOG.md"))

    def test_console_script_target_exists(self):
        target = re.search(r'^backtestlint = "([^"]+)"', read("pyproject.toml"), re.M).group(1)
        module, func = target.split(":")
        self.assertTrue(callable(getattr(__import__(module, fromlist=[func]), func)))

    def test_workflow_is_pinned_and_least_privilege(self):
        wf = read(".github", "workflows", "tests.yml")
        self.assertIn("permissions:\n  contents: read", wf)
        for ref in re.findall(r"uses: (\S+)", wf):
            self.assertRegex(ref, r"@[0-9a-f]{40}$")
        self.assertNotIn("pull_request_target", wf)

    def test_no_dependencies_are_declared(self):
        self.assertNotRegex(read("pyproject.toml"), r"(?m)^dependencies\s*=")

    def test_examples_contain_no_private_paths(self):
        for name in os.listdir(os.path.join(ROOT, "examples")):
            text = read("examples", name)
            self.assertFalse(os.path.expanduser("~") in text or "/Users/" in text, name)

    def test_readme_documents_every_rule(self):
        readme = read("README.md")
        for rid in RULES:
            self.assertIn(rid, readme)
        self.assertIn("%d rules" % len(RULES), readme)

    def test_leaky_example_triggers_most_rules(self):
        from backtestlint.linter import lint_file
        found = {f.rule_id for f in lint_file(os.path.join(ROOT, "examples", "leaky_backtest.py"))}
        self.assertGreaterEqual(len(found), 7)
        self.assertTrue({"BL001", "BL002", "BL005", "BL007", "BL008"} <= found)

if __name__ == "__main__":
    unittest.main()
