#!/usr/bin/env python3
"""Improved lon/lat plotting for cubed-sphere path variables using xarray + matplotlib."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cubed_sphere_plot_utils import (
    TARGET_VARS,
    equirectangular_sample,
    extract_local_face_coords,
    get_data_arrays,
    load_dataset,
    percentile_range,
    split_mosaic_to_faces,
    write_metadata,
)


def save_lonlat_plot(field, lon_deg, lat_deg, var_name: str, nc_stem: str, outdir: Path):
    vmin, vmax = percentile_range(field, 1.0, 99.0)
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    pcm = ax.pcolormesh(lon_deg, lat_deg, field, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, color="w", alpha=0.25, linewidth=0.5)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"{var_name} (spherical lon/lat projection)\n{nc_stem}")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.92, pad=0.02)
    cb.set_label("kg m$^{-2}$")

    png_path = outdir / f"{nc_stem}.{var_name}.lonlat.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    stats_path = outdir / f"{nc_stem}.{var_name}.stats.txt"
    vals = field[np.isfinite(field)]
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"variable: {var_name}\n")
        f.write(f"projected_shape: {field.shape}\n")
        f.write(f"projected_min: {float(np.min(vals))}\n")
        f.write(f"projected_max: {float(np.max(vals))}\n")
        f.write(f"display_vmin_p1: {vmin}\n")
        f.write(f"display_vmax_p99: {vmax}\n")
        f.write(f"output_image: {png_path.name}\n")
    print(f"Wrote {png_path}")
    print(f"Wrote {stats_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cubed-sphere path variables in lon/lat")
    ap.add_argument("nc_file")
    ap.add_argument("--outdir", default="sub_neptune/plots")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.nc_file)
    raw = get_data_arrays(ds)
    alpha_c, beta_c, alpha_f, beta_f = extract_local_face_coords(raw["x2"], raw["x3"], raw["x2f"], raw["x3f"])
    write_metadata(outdir, alpha_c, beta_c, alpha_f, beta_f)

    nc_stem = Path(args.nc_file).stem
    for var in TARGET_VARS:
        faces = split_mosaic_to_faces(raw[var])
        field, lon_deg, lat_deg = equirectangular_sample(faces, alpha_f, beta_f, width=args.width, height=args.height)
        save_lonlat_plot(field, lon_deg, lat_deg, var, nc_stem, outdir)

    ds.close()


if __name__ == "__main__":
    main()
