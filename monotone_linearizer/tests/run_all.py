"""Run every monotone_linearizer test."""

from __future__ import annotations

import importlib
import sys
import traceback

TEST_MODULES = [
    "monotone_linearizer.tests.test_monotonicity",
    "monotone_linearizer.tests.test_fixed_point",
    "monotone_linearizer.tests.test_invertibility",
    "monotone_linearizer.tests.test_implicit_diff",
    "monotone_linearizer.tests.test_end_to_end",
    "monotone_linearizer.tests.test_overfit_one_batch",
]


def main() -> int:
    failures = []
    for mod_name in TEST_MODULES:
        print(f"\n[ {mod_name} ]")
        try:
            mod = importlib.import_module(mod_name)
            for attr in sorted(dir(mod)):
                if attr.startswith("test_"):
                    fn = getattr(mod, attr)
                    if callable(fn):
                        print(f"  -> {attr} ... ", end="", flush=True)
                        try:
                            fn()
                            print("ok")
                        except Exception as e:
                            print(f"FAIL: {e}")
                            failures.append((mod_name, attr, traceback.format_exc()))
        except Exception as e:
            print(f"  import error: {e}")
            failures.append((mod_name, "<import>", traceback.format_exc()))
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)})")
        for mod, attr, tb in failures:
            print(f"\n--- {mod}::{attr} ---\n{tb}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
