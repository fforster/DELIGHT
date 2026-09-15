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

# stage, then (quantity, tolerance, unit) for each of its quantities.
#
# Tolerances are absolute, applied to the maximum absolute difference, and each
# one is in the unit of its own quantity -- they are not comparable across
# stages.  The images are PanSTARRS cutouts at 0.25 arcsec per pixel
# (Delight.pixscale), so 1e-3 px is 0.25 mas, 1e-4 px is 25 uas, 1e-7 deg is
# 0.36 mas and 1e-3 arcsec is 1 mas.
#
# The bands differ by stage on purpose: the network is expected to agree to
# float32 noise, while the sep- and WCS-derived quantities may move slightly
# with their library versions.
STAGES = [
    # min-max normalised intensity in [0, 1], so dimensionless. Compared
    # exactly: these stages are pure numpy and xarray and should not move at all
    ("preprocessing", [("X", 0.0, ""),
                       ("Xpr", 0.0, "")]),
    # pixel coordinates and offsets on the 480x480 cutout
    ("source extraction", [("xSN", 1e-3, "px"), ("ySN", 1e-3, "px"),
                           ("dx_sex", 1e-3, "px"), ("dy_sex", 1e-3, "px")]),
    # the network predicts offsets in cutout pixels: dx_delight is added to
    # xSN before wcs.pixel_to_world. std_delight is an rms over those offsets
    ("neural network", [("y_raw", 1e-4, "px"), ("dxdy_rotflip", 1e-4, "px"),
                        ("dx_delight", 1e-4, "px"), ("dy_delight", 1e-4, "px"),
                        ("std_delight", 1e-4, "px")]),
    ("coordinates", [("ra_delight", 1e-7, "deg"), ("dec_delight", 1e-7, "deg"),
                     ("ra_sex", 1e-7, "deg"), ("dec_sex", 1e-7, "deg")]),
    # get_hostsize multiplies both of these by pixscale before storing them
    ("host size", [("hostsize", 1e-3, "arcsec"),
                   ("hostsep", 1e-3, "arcsec")]),
    # separation divided by semi-major axis: a ratio, not an angle, so it is
    # kept out of the arcsec group above rather than implying a unit it lacks
    ("host size ratio", [("mindistsize", 1e-3, "")]),
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

    print("%-18s %-16s %12s %12s %-7s %s" % ("stage", "quantity", "max abs diff",
                                             "tolerance", "unit", "worst object"))
    print("-" * 86)

    failures = []
    for stage, quantities in STAGES:
        for name, tol, unit in quantities:
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
                failures.append("%s: max abs diff %.3e > %.1e %s"
                                % (name, maxdiff, tol, unit or "(dimensionless)"))
            print("%-18s %-16s %12.3e %12.1e %-7s %s%s"
                  % (stage, name, maxdiff, tol, unit or "-", oids[worst], status))
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
