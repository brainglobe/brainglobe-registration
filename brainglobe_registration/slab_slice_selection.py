import napari
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from tifffile import tifffile

from brainglobe_registration.automated_target_selection import (
    run_bayesian_generator,
)

atlas = BrainGlobeAtlas("allen_mouse_100um")
atlas_volume = atlas.reference
atlas_res = atlas.resolution

slab = tifffile.imread(
    r"C:\Users\saara\Documents\BrainGlobe 2025\slab_100.tiff"
)

slab_shape = slab.shape  # (10, 85, 108)
# 10 slices, height 85, width 108

slab_resolution = (100, 100, 100)  # fix this

first_slice = slab[0]
last_slice = slab[-1]


def run_registration(atlas_image, moving_image, params):
    result_generator = run_bayesian_generator(
        atlas_image,
        moving_image,
        params["z_range"],
        params["pitch_bounds"],
        params["yaw_bounds"],
        params["roll_bounds"],
        params["init_points"],
        params["n_iter"],
        params["metric"],
        params["weights"],
    )

    try:
        while True:
            next(result_generator)
    except StopIteration as stop:
        final_result = stop.value

    return {
        "done": True,
        "best_pitch": final_result["best_pitch"],
        "best_yaw": final_result["best_yaw"],
        "best_roll": final_result["best_roll"],
        "best_z_slice": final_result["best_z_slice"],
    }


params = {
    "z_range": None,
    "pitch_bounds": (-10, 10),
    "yaw_bounds": (-10, 10),
    "roll_bounds": (-14, -6),
    "init_points": 5,
    "n_iter": 15,
    "metric": "mi",
    "weights": (0.7, 0.15, 0.15),
}

first_result = run_registration(atlas_volume, first_slice, params)
last_result = run_registration(atlas_volume, last_slice, params)

print("First slice result:", first_result)
print("Last slice result:", last_result)

first_z = first_result["best_z_slice"]
last_z = last_result["best_z_slice"]


N = slab.shape[0]
target_depth = last_z - first_z + 1

if target_depth < N:
    print("Case 1: Expanding outward to match number of slab slices")
    current_first = first_z
    current_last = last_z
    while (current_last - current_first + 1) < N:
        # Expand outward symmetrically
        if current_first > 0:
            current_first -= 1
        if (current_last < atlas_volume.shape[0] - 1) and (
            current_last - current_first + 1
        ) < N:
            current_last += 1
    # Define evenly spaced target_z_indices
    target_z_indices = list(range(current_first, current_last + 1))

elif target_depth == N:
    print("Case 2: Exact match, no interpolation or expansion needed")
    target_z_indices = list(range(first_z, last_z + 1))

else:
    print("Case 3: Spacing slab slices across wider range")
    target_z_indices = np.linspace(first_z, last_z, N).astype(int)

print(f"Target z-indices for slab: {target_z_indices}")


# -------- Embed slab into blank volume --------
blank_volume = np.zeros_like(atlas_volume)

# Center slab in XY
sy, sx = slab.shape[1:]
atlas_y, atlas_x = atlas_volume.shape[1:]
y_offset = (atlas_y - sy) // 2
x_offset = (atlas_x - sx) // 2

for slab_idx, atlas_z in enumerate(target_z_indices):
    slice_data = slab[slab_idx]

    if 0 <= atlas_z < blank_volume.shape[0]:
        # Compute offsets
        y_offset = (atlas_volume.shape[1] - slice_data.shape[0]) // 2
        x_offset = (atlas_volume.shape[2] - slice_data.shape[1]) // 2

        # Compute target Y and X ranges
        y_start = max(0, y_offset)
        x_start = max(0, x_offset)
        y_end = min(y_start + slice_data.shape[0], atlas_volume.shape[1])
        x_end = min(x_start + slice_data.shape[1], atlas_volume.shape[2])

        # Compute slice crop range in case it's larger than the target area
        y_slice_end = y_end - y_start
        x_slice_end = x_end - x_start

        # Embed cropped slice safely
        blank_volume[atlas_z, y_start:y_end, x_start:x_end] = slice_data[
            :y_slice_end, :x_slice_end
        ]

# -------- View in napari --------
viewer = napari.Viewer()
viewer.add_image(atlas_volume, name="atlas")
viewer.add_image(blank_volume, name="embedded slab")
napari.run()
