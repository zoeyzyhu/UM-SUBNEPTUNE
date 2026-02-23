#!/usr/bin/env python3
"""Orthographic projection plots for cubed-sphere path variables using matplotlib."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def _draw_orthographic_figure(field, var_name: str, nc_stem: str, center_lon: float, center_lat: float, vmin: float, vmax: float):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)

    masked = np.ma.masked_invalid(field)
    im = ax.imshow(masked, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax, extent=(-1, 1, -1, 1))

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
    return fig


def save_orthographic_plot(field, visible, var_name: str, nc_stem: str, outdir: Path, center_lon: float, center_lat: float, vmin: float | None = None, vmax: float | None = None):
    if vmin is None or vmax is None:
        vmin, vmax = percentile_range(field, 1.0, 99.0)
    fig = _draw_orthographic_figure(field, var_name, nc_stem, center_lon, center_lat, vmin, vmax)

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


def save_orthographic_spin_gif(
    faces: np.ndarray,
    alpha_f: np.ndarray,
    beta_f: np.ndarray,
    var_name: str,
    nc_stem: str,
    outdir: Path,
    center_lat: float,
    start_lon: float,
    nframes: int,
    fps: float,
    vmin: float,
    vmax: float,
    size: int,
):
    frames: list[Image.Image] = []
    longitudes = np.linspace(start_lon, start_lon + 360.0, nframes, endpoint=False)
    for lon in longitudes:
        field, visible = orthographic_sample(
            faces,
            alpha_f,
            beta_f,
            center_lon_deg=float(lon),
            center_lat_deg=center_lat,
            size=size,
        )
        fig = _draw_orthographic_figure(field, var_name, nc_stem, float(lon), center_lat, vmin, vmax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE))

    slug = f"spin_lat{center_lat:+.0f}_lon{start_lon:+.0f}".replace("+", "p").replace("-", "m")
    gif_path = outdir / f"{nc_stem}.{var_name}.ortho.{slug}.gif"
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Wrote {gif_path}")

    stats_path = outdir / f"{nc_stem}.{var_name}.ortho.{slug}.stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"variable: {var_name}\n")
        f.write("projection: orthographic_spin_gif\n")
        f.write(f"center_lat_deg: {center_lat}\n")
        f.write(f"start_lon_deg: {start_lon}\n")
        f.write(f"nframes: {nframes}\n")
        f.write(f"fps: {fps}\n")
        f.write(f"display_vmin_p1: {vmin}\n")
        f.write(f"display_vmax_p99: {vmax}\n")
        f.write(f"output_gif: {gif_path.name}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orthographic projection plots for cubed-sphere path variables")
    ap.add_argument("nc_file")
    ap.add_argument("--outdir", default="sub_neptune/plots_ortho")
    ap.add_argument("--size", type=int, default=900, help="Image raster size in pixels")
    ap.add_argument("--center-lon", type=float, default=0.0)
    ap.add_argument("--center-lat", type=float, default=20.0)
    ap.add_argument("--spin-gif", action="store_true", help="Generate spinning GIFs (one per variable)")
    ap.add_argument("--spin-frames", type=int, default=36, help="Number of frames in spinning GIF")
    ap.add_argument("--spin-fps", type=float, default=10.0, help="GIF frames per second")
    ap.add_argument("--spin-start-lon", type=float, default=0.0, help="Starting center longitude for spin")
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
        vmin, vmax = percentile_range(raw[var], 1.0, 99.0)
        field, visible = orthographic_sample(
            faces,
            alpha_f,
            beta_f,
            center_lon_deg=args.center_lon,
            center_lat_deg=args.center_lat,
            size=args.size,
        )
        save_orthographic_plot(field, visible, var, nc_stem, outdir, args.center_lon, args.center_lat, vmin=vmin, vmax=vmax)
        if args.spin_gif:
            save_orthographic_spin_gif(
                faces=faces,
                alpha_f=alpha_f,
                beta_f=beta_f,
                var_name=var,
                nc_stem=nc_stem,
                outdir=outdir,
                center_lat=args.center_lat,
                start_lon=args.spin_start_lon,
                nframes=args.spin_frames,
                fps=args.spin_fps,
                vmin=vmin,
                vmax=vmax,
                size=args.size,
            )

    ds.close()


if __name__ == "__main__":
    main()
