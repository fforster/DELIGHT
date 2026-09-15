"""Run the whole DELIGHT pipeline and dump the results as a flat npz.

The same script runs under both the legacy (TensorFlow 2.15 / Keras 2) and the
modern (TensorFlow 2.21 / Keras 3) environment, so the two runs can be compared
element by element.  Keep it Python 3.9 compatible for that reason.

Results are written as plain arrays rather than a pickled DataFrame: the
DataFrame stores astropy WCS and SkyCoord objects in its cells, which do not
compare cleanly across astropy and pandas versions.

    python tools/run_pipeline.py --datadir data --out artifacts/golden_new.npz

Nothing is downloaded; the run uses whatever fits files are already in
<datadir>/fits, so it works offline.
"""

import argparse
import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# scalar columns copied straight out of the resulting dataframe
SCALARS = ["ra", "dec", "dist", "xSN", "ySN", "dx", "dy",
           "dx_sex", "dy_sex", "dx_delight", "dy_delight", "std_delight",
           "ra_delight", "dec_delight", "ra_sex", "dec_sex",
           "hostsize", "hostsep", "mindistsize"]


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datadir", default="data")
    parser.add_argument("--coords", default=None,
                        help="csv with oid,ra,dec (default <datadir>/testcoords.csv)")
    parser.add_argument("--out", required=True, help="output npz")
    parser.add_argument("--xpr-out", default=None,
                        help="optional npz holding just the preprocessed model input")
    parser.add_argument("--nlevels", type=int, default=5)
    parser.add_argument("--package", default=None,
                        help="directory to prepend to sys.path, to pick a "
                             "particular checkout of the delight package")
    args = parser.parse_args()

    if args.package:
        sys.path.insert(0, args.package)

    from delight.delight import Delight

    import keras
    print("python %s" % sys.version.split()[0])
    print("keras %s, numpy %s, pandas %s" % (keras.__version__, np.__version__, pd.__version__))

    coords = args.coords or os.path.join(args.datadir, "testcoords.csv")
    df = pd.read_csv(coords)
    print("%i objects from %s" % (len(df), coords))

    client = Delight(args.datadir, df.oid.values, df.ra.values, df.dec.values)

    # no download(): the run is offline and uses the fits already on disk
    if not client.check_missing():
        raise SystemExit("no fits files in %s" % client.downloadfolder)
    missing = (client.df.dist > 0.1).sum()
    if missing:
        raise SystemExit("%i objects have no local fits file" % missing)

    client.get_pix_coords()
    client.compute_multiresolution(args.nlevels, False, True, False)
    client.load_model()
    client.preprocess()
    client.predict()
    for oid in client.df.index:
        client.get_hostsize(oid, doplot=False)

    # the raw network output, before derotate() undoes the rotations and flips.
    # The pristine 2022 code does not keep it, so recompute it here rather than
    # patching that checkout.
    if hasattr(client, "y_pred_raw"):
        y_raw = client.y_pred_raw
    else:
        y_raw = client.tfmodel.predict(
            [client.Xpr[:, :, :, i] for i in range(client.Xpr.shape[3])])

    out = {"oids": np.asarray(client.oids, dtype="U32"),
           "X": np.asarray(client.X, dtype="float64"),
           "Xpr": np.asarray(client.Xpr, dtype="float64"),
           "y_raw": np.asarray(y_raw, dtype="float64"),
           "dxdy_rotflip": np.stack(client.df["dxdy_delight_rotflip"].to_numpy()).astype("float64")}

    for name in SCALARS:
        if name in client.df:
            out[name] = client.df[name].to_numpy(dtype="float64")
        else:
            print("   WARNING: column %s missing" % name)

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote %s" % args.out)
    for key in sorted(out):
        print("   %-16s %s" % (key, np.shape(out[key])))

    if args.xpr_out:
        outdir = os.path.dirname(args.xpr_out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(args.xpr_out, Xpr=np.asarray(client.Xpr, dtype="float32"))
        print("wrote %s" % args.xpr_out)

    # also refresh the pickled dataframe, so it stops being a one-off artifact
    client.save()


if __name__ == "__main__":
    main()
