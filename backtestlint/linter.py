"""AST-based detection of look-ahead and leakage patterns. Pure standard library."""
import ast
import json
import os
import re

from .rules import RULES

STAT_METHODS = {"mean", "std", "max", "min", "median", "quantile", "var"}
SIGNAL_WORDS = ("sig", "position", "pos", "weight", "side", "target", "holding", "exposure")
RETURN_WORDS = ("ret", "return", "pnl", "pct")
FIT_OBJECT_WORDS = ("scaler", "normalizer", "pca", "imputer", "encoder", "transformer", "standardizer")
SPLIT_FUNCS = {"train_test_split"}
IGNORE_RE = re.compile(r"#\s*backtestlint:\s*ignore(?:\[([A-Za-z0-9_, ]+)\])?")
IGNORE_FILE_RE = re.compile(r"^\s*#\s*backtestlint:\s*ignore-file", re.M)


class Finding:
    def __init__(self, rule_id, line, col, message, confidence="MEDIUM", path="<string>", severity=None):
        self.rule_id, self.line, self.col, self.message, self.confidence, self.path = rule_id, line, col, message, confidence, path
        self.severity = severity or RULES[rule_id].severity

    def to_dict(self):
        r = RULES[self.rule_id]
        return {"rule": self.rule_id, "name": r.name, "severity": self.severity, "confidence": self.confidence, "path": self.path,
                "line": self.line, "column": self.col, "message": self.message, "fix": r.fix}


def const_int(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = const_int(node.operand)
        return -inner if inner is not None else None
    return None


def kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def method(call):
    """Method or function name of a call: x.shift(1) -> 'shift', f(x) -> 'f'."""
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def receiver_name(call):
    f = call.func
    if isinstance(f, ast.Attribute):
        v = f.value
        if isinstance(v, ast.Name):
            return v.id
        if isinstance(v, ast.Attribute):
            return v.attr
    return None


def name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return name_of(node.value)
    return None


def root_name(node):
    """df.close.iloc -> 'df'"""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if not isinstance(node, ast.Call) else node.func
    return node.id if isinstance(node, ast.Name) else None


def has_word(name, words):
    low = (name or "").lower()
    return any(w in low for w in words)


def chain_has(node, methods):
    """True when a method chain below `node` calls one of `methods` (e.g. x.rolling(5).mean())."""
    while isinstance(node, (ast.Call, ast.Attribute)):
        if isinstance(node, ast.Call):
            if method(node) in methods:
                return True
            node = node.func
        else:
            node = node.value
    return False


class Visitor(ast.NodeVisitor):
    def __init__(self, tree):
        self.findings = []
        self.loops = []                       # stack of (loop_var, iterated_name)
        self.shifted = self._shifted_names(tree)
        self.has_split = any(isinstance(n, ast.Call) and method(n) in SPLIT_FUNCS for n in ast.walk(tree))
        self.fit_calls = []

    @staticmethod
    def _shifted_names(tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call) and method(sub) == "shift":
                        arg = sub.args[0] if sub.args else kwarg(sub, "periods")
                        n = const_int(arg) if arg is not None else 1
                        if n is None or n > 0:
                            for t in node.targets:
                                if name_of(t):
                                    names.add(name_of(t))
        return names

    def add(self, rule, node, message, confidence="MEDIUM"):
        self.findings.append(Finding(rule, node.lineno, node.col_offset, message, confidence))

    # -- loops -----------------------------------------------------------------
    def visit_For(self, node):
        var = node.target.id if isinstance(node.target, ast.Name) else None
        iterated = None
        it = node.iter
        if isinstance(it, ast.Call) and method(it) == "range" and it.args:
            for a in ast.walk(it):
                if isinstance(a, ast.Call) and method(a) == "len" and a.args:
                    iterated = root_name(a.args[0])
        elif isinstance(it, ast.Call) and method(it) in ("iterrows", "itertuples", "items"):
            iterated = root_name(it.func.value)
        elif isinstance(it, ast.Attribute) and it.attr == "index":
            iterated = root_name(it.value)
        self.loops.append((var, iterated))
        self.generic_visit(node)
        self.loops.pop()

    # -- subscripts ---------------------------------------------------------------
    def visit_Subscript(self, node):
        sl = node.slice
        base = node.value
        is_iloc = isinstance(base, ast.Attribute) and base.attr in ("iloc", "loc")
        if self.loops:
            var = self.loops[-1][0]
            if isinstance(sl, ast.BinOp) and isinstance(sl.op, ast.Add) and isinstance(sl.left, ast.Name) and sl.left.id == var:
                k = const_int(sl.right)
                if k is not None and k > 0:
                    self.add("BL003", node, "%s[%s + %d] reads a later row than the current iteration" % (name_of(base) or "data", var, k),
                             "HIGH" if is_iloc else "MEDIUM")
            if const_int(sl) == -1 and any(root_name(base) == it for _v, it in self.loops if it):
                self.add("BL012", node, "[-1] on %s inside a loop over its own rows is the last bar of the whole dataset" % root_name(base), "MEDIUM")
        self.generic_visit(node)

    # -- binary expressions --------------------------------------------------------
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult):
            left, right = node.left, node.right
            for sig, ret in ((left, right), (right, left)):
                sname, rname = name_of(sig), name_of(ret)
                ret_is_pct = isinstance(ret, ast.Call) and method(ret) == "pct_change"
                if (has_word(sname, SIGNAL_WORDS) and (has_word(rname, RETURN_WORDS) or ret_is_pct)
                        and not isinstance(sig, ast.Call) and sname not in self.shifted
                        and not (rname and has_word(rname, SIGNAL_WORDS))):
                    self.add("BL004", node, "'%s' is multiplied by '%s' but never shifted; this trades on same-bar information" % (sname, rname or "returns"), "LOW")
                    break
        if isinstance(node.op, (ast.Sub, ast.Div)):
            self._full_sample(node)
        self.generic_visit(node)

    def _full_sample(self, node):
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and method(call) in STAT_METHODS and isinstance(call.func, ast.Attribute):
                target = call.func.value
                if not isinstance(target, (ast.Name, ast.Attribute)) or chain_has(target, {"rolling", "expanding", "ewm", "groupby"}):
                    continue
                key = ast.unparse(target)
                if any(w in key.lower() for w in ("train", "fit_")):
                    continue
                inside = {id(n) for n in ast.walk(call)}
                if any(id(n) not in inside and isinstance(n, (ast.Name, ast.Attribute)) and ast.unparse(n) == key for n in ast.walk(node)):
                    self.add("BL007", call, "'%s' is combined with its own whole-sample %s()" % (key, method(call)), "MEDIUM")
                    return

    # -- calls ----------------------------------------------------------------------
    def visit_Call(self, node):
        m = method(node)
        if m == "shift":
            arg = node.args[0] if node.args else kwarg(node, "periods")
            n = const_int(arg) if arg is not None else None
            if n is not None and n < 0:
                self.add("BL001", node, "shift(%d) pulls values from %d row(s) in the future" % (n, -n), "HIGH")
        elif m == "roll" and len(node.args) >= 2 and (name_of(node.func.value) if isinstance(node.func, ast.Attribute) else None) in ("np", "numpy"):
            n = const_int(node.args[1])
            if n is not None and n < 0:
                self.add("BL001", node, "np.roll with a negative shift pulls values from the future", "MEDIUM")
        elif m == "rolling":
            c = kwarg(node, "center")
            if isinstance(c, ast.Constant) and c.value is True:
                self.add("BL002", node, "rolling(center=True) averages rows before and after each point", "HIGH")
        elif m in ("bfill", "backfill"):
            self.add("BL008", node, "%s() fills gaps with the next known value" % m, "HIGH")
        elif m in ("fillna", "interpolate", "ffill"):
            meth = kwarg(node, "method")
            if isinstance(meth, ast.Constant) and meth.value in ("bfill", "backfill", "nearest"):
                self.add("BL008", node, "method=%r fills gaps with future values" % meth.value, "HIGH")
            elif m == "interpolate" and meth is None and not kwarg(node, "limit_direction"):
                self.add("BL008", node, "interpolate() linearly mixes the previous and the next known value", "LOW")
        elif m == "merge_asof":
            d = kwarg(node, "direction")
            if isinstance(d, ast.Constant) and d.value in ("forward", "nearest"):
                self.add("BL009", node, "merge_asof(direction=%r) joins later observations to earlier rows" % d.value, "HIGH")
        elif m == "dropna":
            ax = kwarg(node, "axis")
            if (isinstance(ax, ast.Constant) and ax.value in (1, "columns")) or (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in (1, "columns")):
                self.add("BL010", node, "dropna(axis=1) deletes assets that have gaps (typically the delisted ones)", "MEDIUM")
        elif m in ("pct_change", "diff"):
            arg = node.args[0] if node.args else kwarg(node, "periods")
            n = const_int(arg) if arg is not None else None
            if n is not None and n < 0:
                self.add("BL011", node, "%s(%d) is a forward-looking change" % (m, n), "HIGH")
        elif m in ("qcut", "rank") and not chain_has(node.func, {"groupby", "rolling", "expanding"}):
            if m == "qcut" or (m == "rank" and isinstance(kwarg(node, "pct"), ast.Constant) and kwarg(node, "pct").value is True):
                self.add("BL013", node, "%s() is computed over the whole sample" % m, "LOW")
        elif m == "train_test_split":
            s = kwarg(node, "shuffle")
            if not (isinstance(s, ast.Constant) and s.value is False):
                self.add("BL005", node, "train_test_split shuffles by default" if s is None else "train_test_split(shuffle=True)", "HIGH" if s is None else "HIGH")
        elif m in ("KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "GroupKFold"):
            sh = kwarg(node, "shuffle")
            shuffled = m.endswith("ShuffleSplit") or (isinstance(sh, ast.Constant) and sh.value is True)
            self.add("BL005" if shuffled else "BL014", node, "%s%s on ordered data" % (m, " with shuffling" if shuffled else ""), "MEDIUM" if shuffled else "LOW")
        elif m in ("fit", "fit_transform") and isinstance(node.func, ast.Attribute):
            rname = name_of(node.func.value)
            arg = name_of(node.args[0]) if node.args else None
            if has_word(rname, FIT_OBJECT_WORDS) or (isinstance(node.func.value, ast.Call) and has_word(method(node.func.value), FIT_OBJECT_WORDS)):
                if not has_word(arg, ("train", "fit", "tr_")):
                    self.fit_calls.append((node, arg))
        self.generic_visit(node)

    def finish(self):
        if self.has_split:
            for node, arg in self.fit_calls:
                self.add("BL006", node, "transformer is fitted on '%s' in a script that also splits the data" % (arg or "the data"), "LOW")


def lint_source(source, path="<string>"):
    """Lint Python source text. Returns a list of Finding (already filtered by inline suppressions)."""
    if IGNORE_FILE_RE.search(source):
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise SyntaxError("%s: %s" % (path, exc)) from exc
    v = Visitor(tree)
    v.visit(tree)
    v.finish()
    lines = source.splitlines()
    out, seen = [], set()
    for f in sorted(v.findings, key=lambda x: (x.line, x.col, x.rule_id)):
        if (f.rule_id, f.line) in seen:                       # one report per rule and line
            continue
        seen.add((f.rule_id, f.line))
        f.path = path
        text = lines[f.line - 1] if 0 < f.line <= len(lines) else ""
        m = IGNORE_RE.search(text)
        if m and (m.group(1) is None or f.rule_id in [r.strip() for r in m.group(1).split(",")]):
            continue
        out.append(f)
    return out


def notebook_source(text):
    """Concatenate the code cells of a Jupyter notebook; returns (source, {line number: cell number})."""
    nb = json.loads(text)
    parts, cellmap, line = [], {}, 0
    for i, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        lines = ["" if l.lstrip().startswith(("%", "!")) else l for l in src.rstrip("\n").split("\n")]   # drop IPython magics
        for _ in lines:
            line += 1
            cellmap[line] = i
        parts.append("\n".join(lines))
    return "\n".join(parts), cellmap


def lint_file(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if path.endswith(".ipynb"):
        source, cellmap = notebook_source(text)
        found = lint_source(source, path)
        for f in found:
            f.message += " (cell %d)" % cellmap.get(f.line, 0)
        return found
    return lint_source(text, path)


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ipynb_checkpoints", "build", "dist", ".tox"}


def collect_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for f in sorted(files):
                    if f.endswith((".py", ".ipynb")):
                        yield os.path.join(root, f)
        else:
            yield p
