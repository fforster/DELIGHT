"""Run the legacy Keras 2 DELIGHT model on a saved input tensor.

Used to produce the old-version reference output for the migration's
equivalence check.  Must be run under a Keras 2 environment (TensorFlow 2.15),
where the original ``DELIGHT_v1.h5`` still deserializes.

    python tools/legacy_predict.py --input artifacts/Xpr_golden.npz \
                                   --output artifacts/y_raw_legacy.npz
"""

import argparse
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import tensorflow as tf


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="delight/delight/DELIGHT_v1.h5")
    parser.add_argument("--input", required=True,
                        help="npz holding Xpr with shape (n, 30, 30, nlevels)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    import keras
    print("tensorflow %s, keras %s" % (tf.__version__, keras.__version__))

    Xpr = np.load(args.input)["Xpr"]
    print("input %s" % (Xpr.shape,))

    model = tf.keras.models.load_model(args.model)
    inputs = [Xpr[:, :, :, i][..., np.newaxis].astype("float32")
              for i in range(Xpr.shape[3])]
    y_raw = model.predict(inputs, verbose=0)
    print("output %s" % (y_raw.shape,))

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(args.output, y_raw=y_raw)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
