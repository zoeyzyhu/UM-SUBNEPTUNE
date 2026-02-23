#!/usr/bin/env python3
"""Orthographic projection plots for cubed-sphere path variables using matplotlib."""

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
    extract_local_face_coords,
    get_data_arrays,
    load_dataset,
    orthographic_sample,
    percentile_range,
    split_mosaic_to_faces,
    write_metadata,
)


def save_orthographic_plot(field, visible, var_name: str, nc_stem: str, outdir: Path, center_lon: float, center_lat: float):
    vmin, vmax = percentile_range(field, 1.0, 99.0)
    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)

    masked = np.ma.masked_invalid(field)
    im = ax.imshow(masked, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=(-1, 1, -1, 1))

    # limb
    t = np.linspace(0, 2 * np.pi, 720)
    ax.plot(np.cos(t), np.sin(t), color="white", lw=1.0, alpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    ax.set_title(
        f"{var_name} (orthographic)\ncenter lon={center_lon:.1f}°, lat={center_lat:.1f}°\n{nc_stem}"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("kg m$^{-2}$")

    slug = f"lon{center_lon:+.0f}_lat{center_lat:+.0f}".replace('+','p').replace('-','m')
    png_path = outdir / f"{nc_stem}.{var_name}.ortho.{slug}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    stats_path = outdir / f"{nc_stem}.{var_name}.ortho.{slug}.stats.txt"
    vals = field[np.isfinite(field)]
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"variable: {var_name}\n")
        f.write(f"projection: orthographic\n")
        f.write(f"center_lon_deg: {center_lon}\n")
        f.write(f"center_lat_deg: {center_lat}\n")
        f.write(f"projected_shape: {field.shape}\n")
        f.write(f"projected_min: {float(np.min(vals))}\n")
        f.write(f"projected_max: {float(np.max(vals))}\n")
        f.write(f"display_vmin_p1: {vmin}\n")
        f.write(f"display_vmax_p99: {vmax}\n")
        f.write(f"output_image: {png_path.name}\n")
    print(f"Wrote {png_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orthographic projection plots for cubed-sphere path variables")
    ap.add_argument("nc_file")
    ap.add_argument("--outdir", default="sub_neptune/plots_ortho")
    ap.add_argument("--size", type=int, default=900, help="Image raster size in pixels")
    ap.add_argument("--center-lon", type=float, default=0.0)
    ap.add_argument("--center-lat", type=float, default=20.0)
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
        field, visible = orthographic_sample(
            faces,
            alpha_f,
            beta_f,
            center_lon_deg=args.center_lon,
            center_lat_deg=args.center_lat,
            size=args.size,
        )
        save_orthographic_plot(field, visible, var, nc_stem, outdir, args.center_lon, args.center_lat)

    ds.close()


if __name__ == "__main__":
    main()
