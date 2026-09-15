"""Compare two DELIGHT pipeline runs stage by stage.

Reports the largest absolute difference for each quantity and the object it
occurs on, so a discrepancy can be attributed to a stage -- preprocessing,
source extraction, the neural network or the WCS -- rather than blamed on the
model.

    python tools/compare_runs.py artifacts/golden_legacy.npz artifacts/golden_new.npz
"""

import argparse
import sys

import numpy as np

# stage, quantity, tolerance.  The bands differ by stage on purpose:
# the network is expected to agree to float32 noise, while the sep- and
# WCS-derived quantities may move slightly with their library versions.
STAGES = [
    ("preprocessing", [("X", 0.0), ("Xpr", 0.0)]),
    ("source extraction", [("xSN", 1e-3), ("ySN", 1e-3),
                           ("dx_sex", 1e-3), ("dy_sex", 1e-3)]),
    ("neural network", [("y_raw", 1e-4), ("dxdy_rotflip", 1e-4),
                        ("dx_delight", 1e-4), ("dy_delight", 1e-4),
                        ("std_delight", 1e-4)]),
    ("coordinates", [("ra_delight", 1e-7), ("dec_delight", 1e-7),
                     ("ra_sex", 1e-7), ("dec_sex", 1e-7)]),
    ("host size", [("hostsize", 1e-3), ("hostsep", 1e-3),
                   ("mindistsize", 1e-3)]),
]


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()

    old = np.load(args.old)
    new = np.load(args.new)

    oids = old["oids"]
    if not np.array_equal(oids, new["oids"]):
        raise SystemExit("the two runs cover different objects")
    print("comparing %i objects\n" % len(oids))

    print("%-18s %-16s %12s %12s  %s" % ("stage", "quantity", "max abs diff",
                                         "tolerance", "worst object"))
    print("-" * 78)

    failures = []
    for stage, quantities in STAGES:
        for name, tol in quantities:
            if name not in old.files or name not in new.files:
                print("%-18s %-16s %12s" % (stage, name, "MISSING"))
                continue
            a, b = np.asarray(old[name], "float64"), np.asarray(new[name], "float64")
            if a.shape != b.shape:
                failures.append("%s: shape %s vs %s" % (name, a.shape, b.shape))
                continue

            diff = np.abs(a - b)
            # collapse every axis but the object axis to find the worst object
            perobject = diff.reshape(diff.shape[0], -1).max(axis=1) if diff.ndim > 1 else diff
            worst = int(np.argmax(perobject))
            maxdiff = float(diff.max())

            status = "" if maxdiff <= tol else "   <-- EXCEEDS TOLERANCE"
            if maxdiff > tol:
                failures.append("%s: max abs diff %.3e > %.1e" % (name, maxdiff, tol))
            print("%-18s %-16s %12.3e %12.1e  %s%s"
                  % (stage, name, maxdiff, tol, oids[worst], status))
        print()

    if failures:
        print("FAILED:")
        for line in failures:
            print("   %s" % line)
        return 1

    print("all stages agree within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
