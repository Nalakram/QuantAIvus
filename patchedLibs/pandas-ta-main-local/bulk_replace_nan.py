# bulk_replace_nan.py
# A script to normalize NaN usage, remove njit cache, replace pkg_resources, and update deprecated pandas APIs.
import re
from pathlib import Path

# —————————————————————————————————————————————————————
# 1) NaN replacements
#    - from numpy import NaN -> import numpy as np
#    - np.NaN or bare NaN -> np.nan (avoiding variable-name collisions)
# —————————————————————————————————————————————————————
import_re  = re.compile(r'^\s*from numpy import NaN\s*$', flags=re.MULTILINE)
npnan_re   = re.compile(r'\bnp\.NaN\b')
nan_re     = re.compile(r'\bNaN\b(?!\s*\w)')  # avoid NaNVar, "NaN_value", etc.

# —————————————————————————————————————————————————————
# 2) Strip cache=True from @njit decorators
# —————————————————————————————————————————————————————
njit_re = re.compile(r'@njit\((.*?)\)', flags=re.DOTALL)

def strip_cache(match):
    args = match.group(1).split(',')
    kept = [a.strip() for a in args if not re.match(r'\s*cache\s*=\s*True\s*', a)]
    inner = ', '.join([k for k in kept if k])
    return f"@njit({inner})" if inner else "@njit()"

# —————————————————————————————————————————————————————
# 3) Replace pkg_resources version lookup
# —————————————————————————————————————————————————————
pkg_resources_re = re.compile(
    r'from pkg_resources import get_distribution\n'
    r'\s*__version__ = get_distribution\("pandas_ta"\)\.version'
)

# —————————————————————————————————————————————————————
# 4) Deprecated Pandas API replacements
#    Examples:
#      - df.append(...) -> df = pd.concat([df, ...]) (simplified)
#      - df.iteritems() -> df.items()
# —————————————————————————————————————————————————————
append_re      = re.compile(r'\.append\(')
iteritems_re   = re.compile(r'\.iteritems\(')
# (Note: These replacements may need manual review for context.)

# —————————————————————————————————————————————————————
# 5) Scan for other deprecated Pandas APIs
# —————————————————————————————————————————————————————
deprecated = {
    'Index.iteritems()':     re.compile(r'\.iteritems\('),
    'Panel':                 re.compile(r'\bPanel\b'),
    'ix indexing':          re.compile(r'\.ix\['),
}

root = Path(__file__).parent / "pandas_ta"

# Scan first
print("\n--- Deprecated Pandas API scan before replacement ---")
for path in root.rglob("*.py"):
    rel = path.relative_to(root.parent)
    for i, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        for name, pat in deprecated.items():
            if pat.search(line):
                print(f"⚠ {name} in {rel}:{i}: {line.strip()}")

# Apply replacements and write back
print("\n--- Applying replacements ---")
for path in root.rglob("*.py"):
    text = path.read_text(encoding="utf8")
    new  = text

    # 1) NaN logic
    new = import_re.sub("import numpy as np", new)
    new = npnan_re.sub("np.nan", new)
    new = nan_re.sub("np.nan", new)

    # 2) njit cache removal
    new = njit_re.sub(strip_cache, new)

    # 3) pkg_resources replacement in __init__.py
    if path.name == "__init__.py":
        new = pkg_resources_re.sub(
            'from importlib.metadata import version\n__version__ = version("pandas_ta")',
            new
        )

    # 4) append -> concat
    new = append_re.sub('.concat([', new)
    # 4b) iteritems() -> items()
    new = iteritems_re.sub('.items(', new)

    if new != text:
        path.write_text(new, encoding="utf8")
        print(f"✔ Updated {path.relative_to(root.parent)}")

print("\nAll done.")
