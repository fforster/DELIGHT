"""Convert the legacy Keras 2 DELIGHT model to Keras 3 weights.

The distributed ``DELIGHT_v1.h5`` was written by Keras 2.7 and stores its
rotation/flip/concatenate operations as ``TFOpLambda`` layers, which Keras 3
cannot deserialize.  Only five layers in that file actually carry weights, so
this script reads them straight out of the HDF5 with h5py -- no Keras 2
required -- and writes them into the architecture defined in
``delight.delight.model``.

Run once, from the repository root:

    python tools/convert_model.py

Writes ``delight/delight/DELIGHT_v1.weights.h5`` (the file the library loads)
and ``delight/delight/DELIGHT_v1.keras`` (a whole-model file, for convenience).
"""

import argparse
import hashlib
import importlib.util
import os
import sys

import h5py
import numpy as np

# Layer name -> (kernel shape, bias shape) as stored in the legacy file.
# Asserted on load so a silently different source file cannot slip through.
EXPECTED = {
    "conv2d_48": ((3, 3, 1, 52), (52,)),
    "conv2d_49": ((3, 3, 52, 57), (57,)),
    "conv2d_50": ((3, 3, 57, 41), (41,)),
    "dense_32": ((3280, 685), (685,)),
    "dense_33": ((685, 2), (2,)),
}

# The legacy model, for reference: 5 inputs, a (None, 16) output, and this many
# trainable parameters.  The rebuilt model must agree exactly.
EXPECTED_PARAMS = 2297184


def read_legacy_weights(h5file):

    """read the trained weights out of a Keras 2 DELIGHT .h5 file

    Parameters
    ----------
    h5file : string
       path to DELIGHT_v1.h5
    """

    weights = {}
    with h5py.File(h5file, "r") as f:
        version = f.attrs.get("keras_version", b"")
        if isinstance(version, bytes):
            version = version.decode()
        print("source file was written by Keras %s" % version)

        group = f["model_weights"]
        for name, (kshape, bshape) in EXPECTED.items():
            kernel = np.asarray(group["%s/%s/kernel:0" % (name, name)])
            bias = np.asarray(group["%s/%s/bias:0" % (name, name)])
            if kernel.shape != kshape or bias.shape != bshape:
                raise ValueError(
                    "%s has shapes %s/%s, expected %s/%s"
                    % (name, kernel.shape, bias.shape, kshape, bshape))
            weights[name] = [kernel, bias]
            print("   read %-16s kernel %-16s bias %s"
                  % (name, kernel.shape, bias.shape))
    return weights


def sha256(filename):

    """hex digest of a file, recorded so the test suite can detect a swap"""

    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", default="delight/delight/DELIGHT_v1.h5",
                        help="input Keras 2 model file")
    parser.add_argument("--outdir", default="delight/delight",
                        help="directory to write the converted model into")
    parser.add_argument("--version", default="v1",
                        help="model version suffix")
    args = parser.parse_args()

    # import the architecture without importing the whole delight package,
    # which pulls in astropy, sep and friends
    here = os.path.dirname(os.path.abspath(__file__))
    modelpy = os.path.join(here, os.pardir, "delight", "delight", "model.py")
    spec = importlib.util.spec_from_file_location("delight_model", modelpy)
    dmodel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dmodel)

    weights = read_legacy_weights(args.legacy)

    model = dmodel.build_delight_model()

    # structural checks against the legacy graph
    if len(model.inputs) != dmodel.NLEVELS:
        raise ValueError("expected %i inputs, got %i" % (dmodel.NLEVELS, len(model.inputs)))
    if tuple(model.output_shape) != (None, 2 * len(dmodel.VARIANTS)):
        raise ValueError("unexpected output shape %s" % (model.output_shape,))
    if model.count_params() != EXPECTED_PARAMS:
        raise ValueError("expected %i parameters, got %i"
                         % (EXPECTED_PARAMS, model.count_params()))
    print("architecture matches the legacy model: %i inputs, output %s, %i parameters"
          % (len(model.inputs), model.output_shape, model.count_params()))

    for name, values in weights.items():
        layer = model.get_layer(name)
        current = [w.shape for w in layer.get_weights()]
        if current != [v.shape for v in values]:
            raise ValueError("%s expects %s, source has %s"
                             % (name, current, [v.shape for v in values]))
        layer.set_weights(values)
    print("loaded weights into %i layers" % len(weights))

    weightsfile = os.path.join(args.outdir, "DELIGHT_%s.weights.h5" % args.version)
    kerasfile = os.path.join(args.outdir, "DELIGHT_%s.keras" % args.version)
    model.save_weights(weightsfile)
    model.save(kerasfile)

    # the saved files must round-trip, not merely be written
    reloaded = dmodel.load_delight_model(weightsfile)
    probe = [np.random.RandomState(0).rand(2, dmodel.IMSIZE, dmodel.IMSIZE, 1).astype("float32")
             for _ in range(dmodel.NLEVELS)]
    before = model.predict(probe, verbose=0)
    after = reloaded.predict(probe, verbose=0)
    if not np.array_equal(before, after):
        raise ValueError("weights file does not round-trip")

    import keras
    whole = keras.saving.load_model(kerasfile)
    if not np.array_equal(before, whole.predict(probe, verbose=0)):
        raise ValueError("%s does not round-trip" % kerasfile)
    print("both files round-trip exactly")

    for filename in (weightsfile, kerasfile):
        print("   %s  %.1f MB  sha256=%s"
              % (filename, os.path.getsize(filename) / 1e6, sha256(filename)))


if __name__ == "__main__":
    sys.exit(main())
