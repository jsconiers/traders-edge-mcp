"""(#1) Freshness lint — a static guard so the 'which feed am I on?' class of bug (B3/B10/B11)
can't recur silently: every @mcp.tool whose OUTPUT surfaces a `spot` value must also surface
`source` and `freshness`. Pure AST/text, no network. Run standalone or import into the offline suite.
"""
import ast
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "traders_edge_mcp.py"
SRC = open(PATH, encoding="utf-8").read()
TREE = ast.parse(SRC)

_SPOT = re.compile(r'["\']spot["\']\s*:')
_SOURCE = re.compile(r'["\']source["\']\s*:')
_FRESH = re.compile(r'["\']freshness["\']\s*:')


def _is_tool(node) -> bool:
    for d in getattr(node, "decorator_list", []):
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "tool":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "tool":
            return True
    return False


def run(path: str = PATH) -> int:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    checked, fails = 0, []
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and _is_tool(node):
            seg = ast.get_source_segment(src, node) or ""
            if "freshness-lint: exempt" in seg:
                continue
            if _SPOT.search(seg):                       # only tools that emit a spot value
                checked += 1
                ok = bool(_SOURCE.search(seg)) and bool(_FRESH.search(seg))
                if not ok:
                    fails.append((node.name, bool(_SOURCE.search(seg)), bool(_FRESH.search(seg))))
    print(f"freshness lint: checked {checked} spot-returning tools in {path}")
    for nm, has_src, has_fresh in fails:
        print(f"  FAIL {nm}: source={has_src} freshness={has_fresh}")
    if fails:
        print(f"{len(fails)} tool(s) emit spot without source+freshness")
        return 1
    print("FRESHNESS LINT PASSED - every spot-returning tool also returns source + freshness")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
