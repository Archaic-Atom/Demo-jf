import os
import time
import datetime
import argparse

import numpy as np

import rasterio
#import utils

NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,  # 0 No data 0
    0,  # 1 Open Water
    4,  # 2 Ice/Snow
    2,  # 3 Developed Open Space
    3,  # 4 Developed Low Intensity
    3,  # 5 Developed Medium Intensity
    3,  # 6 Developed High Intensity
    3,  # 7 Barren Land
    1,  # 8 Deciduous Forest
    1,  # 9 Evergreen Forest
    1,  # 10 Mixed Forest
    1,  # 11 Shrub/Scrub
    2,  # 12 Grassland/Herbaceous
    2,  # 13 Pasture/Hay
    2,  # 14 Cultivated Crops
    1,  # 15 Woody Wetlands
    1,  # 16 Emergent Herbaceious Wetlands
])

NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR = np.array([
    [0, 0, 0, 0, 1],  # 0 No data 0
    [1, 0, 0, 0, 0],  # 1 Open Water
    [0, 0, 0, 0, 1],  # 2 Ice/Snow
    [0, 0, 0, 0, 0],  # 3 Developed Open Space
    [0, 0, 0, 0, 0],  # 4 Developed Low Intensity
    [0, 0, 0, 1, 0],  # 5 Developed Medium Intensity
    [0, 0, 0, 1, 0],  # 6 Developed High Intensity
    [0, 0, 0, 0, 0],  # 7 Barren Land
    [0, 1, 0, 0, 0],  # 8 Deciduous Forest
    [0, 1, 0, 0, 0],  # 9 Evergreen Forest
    [0, 1, 0, 0, 0],  # 10 Mixed Forest
    [0, 1, 0, 0, 0],  # 11 Shrub/Scrub
    [0, 0, 1, 0, 0],  # 12 Grassland/Herbaceous
    [0, 0, 1, 0, 0],  # 13 Pasture/Hay
    [0, 0, 1, 0, 0],  # 14 Cultivated Crops
    [0, 1, 0, 0, 0],  # 15 Woody Wetlands
    [0, 1, 0, 0, 0],  # 16 Emergent Herbaceious Wetlands
])

parser = argparse.ArgumentParser(
    description='Helper script for combining DFC2021 prediction into submission format')
parser.add_argument('--input_dir', type=str, required=True,
                    help='The path to a directory containing the output of the `inference.py` script.')
parser.add_argument('--output_dir', type=str, required=True,
                    help='The path to output the consolidated predictions, should be different than `--input_dir`.')
parser.add_argument('--overwrite', action="store_true",
                    help='Flag for overwriting `--output_dir` if that directory already exists.')
parser.add_argument('--soft_assignment', action="store_true",
                    help='Flag for combining predictions using soft assignment. You can only use this if you ran the `inference.py` script with the `--save_soft` flag.')
args = parser.parse_args()


def main():
    print("Starting to combine predictions at %s" % (str(datetime.datetime.now())))

    # -------------------
    # Setup
    # -------------------
    assert os.path.exists(args.input_dir) and len(os.listdir(args.input_dir)) > 0

    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        if args.overwrite:
            print("WARNING! The output directory, %s, already exists, we might overwrite data in it!" % (
                args.output_dir))
        else:
            print("The output directory, %s, already exists and isn't empty. We don't want to overwrite and existing results, exiting..." % (
                args.output_dir))
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    # -------------------
    # Run for each pair of predictions that we find in `--input_dir`
    # -------------------
    idxs_2013 = [
        fn.split("_")[0]
        for fn in os.listdir(args.input_dir)
        if fn.endswith("predictions-2013.tif")
    ]

    idxs_2017 = [
        fn.split("_")[0]
        for fn in os.listdir(args.input_dir)
        if fn.endswith("predictions-2017.tif")
    ]

    assert len(idxs_2013) > 0, "No matching files found in '%s'" % (args.input_dir)
    assert set(idxs_2013) == set(idxs_2017), "Missing some predictions"

    for i, idx in enumerate(idxs_2013):
        tic = time.time()

        print("(%d/%d) Processing tile %s" % (i, len(idxs_2013), idx), end=" ... ")

        if args.soft_assignment:
            fn_2013 = os.path.join(args.input_dir, "%s_predictions-soft-2013.tif" % (idx))
            fn_2017 = os.path.join(args.input_dir, "%s_predictions-soft-2017.tif" % (idx))
        else:
            fn_2013 = os.path.join(args.input_dir, "%s_predictions-2013.tif" % (idx))
            fn_2017 = os.path.join(args.input_dir, "%s_predictions-2017.tif" % (idx))
        output_fn = os.path.join(args.output_dir, "%s_predictions.tif" % (idx))

        assert os.path.exists(fn_2013) and os.path.exists(fn_2017)

        # Load the independent predictions for both years
        with rasterio.open(fn_2013) as f:
            t1 = np.rollaxis(f.read(), 0, 3) if args.soft_assignment else f.read(1)
            input_profile = f.profile.copy()  # save the metadata for writing output

        with rasterio.open(fn_2017) as f:
            t2 = np.rollaxis(f.read(), 0, 3) if args.soft_assignment else f.read(1)
        # Convert to reduced land cover predictions
        if args.soft_assignment:
            t1_reduced = (t1 @ NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR).argmax(axis=2)
            t2_reduced = (t2 @ NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR).argmax(axis=2)
        else:
            t1_reduced = NLCD_IDX_TO_REDUCED_LC_MAP[t1]
            t2_reduced = NLCD_IDX_TO_REDUCED_LC_MAP[t2]

        # Convert the two layers of predictions into the format expected by codalab
        predictions = (t1_reduced * 4) + t2_reduced
        predictions[predictions == 5] = 0
        predictions[predictions == 10] = 0
        predictions[predictions == 15] = 0
        predictions = predictions.astype(np.uint8)

        # Write output as GeoTIFF
        input_profile["count"] = 1
        with rasterio.open(output_fn, "w", **input_profile) as f:
            f.write(predictions, 1)

        print("finished in %0.4f seconds" % (time.time() - tic))


if __name__ == "__main__":
    main()
