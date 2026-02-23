#!/usr/bin/env python3
"""Plot cubed-sphere path variables in spherical (lon/lat) coordinates.

This script reads a NetCDF file through `ncdump` (no Python NetCDF dependency),
reconstructs the 6 cubed-sphere faces from the 3x2 NetCDF mosaic layout used by
snapy, projects to spherical coordinates, and writes equirectangular PPM images.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import numpy as np

FACE_NAMES = ["+X", "+Y", "-X", "+Z", "-Y", "-Z"]
TARGET_VARS = ["path_H2O", "path_H2O_l_", "path_H2O_l_p_"]


def run_ncdump(nc_path: str, variables: list[str]) -> str:
    cmd = ["ncdump", "-v", ",".join(variables), nc_path]
    return subprocess.check_output(cmd, text=True)


def _extract_block(text: str, var_name: str) -> str:
    matches = list(re.finditer(rf"\n\s*{re.escape(var_name)}\s*=\s*(.*?);\n", text, re.S))
    if not matches:
        raise KeyError(f"Variable '{var_name}' not found in ncdump output")
    return matches[-1].group(1)


def _parse_float_block(block: str) -> np.ndarray:
    tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[Ee][-+]?\d+)?|NaN|nan", block)
    out = np.empty(len(tokens), dtype=np.float64)
    for i, tok in enumerate(tokens):
        if tok.lower() == "nan":
            out[i] = np.nan
        else:
            out[i] = float(tok)
    return out


def load_arrays_via_ncdump(nc_path: str) -> dict[str, np.ndarray]:
    vars_to_dump = ["x2", "x2f", "x3", "x3f", *TARGET_VARS]
    text = run_ncdump(nc_path, vars_to_dump)
    data = {}
    for name in vars_to_dump:
        data[name] = _parse_float_block(_extract_block(text, name))

    # Infer shapes from coordinate lengths and known variable dimensions (time, x3, x2)
    nx2 = data["x2"].size
    nx3 = data["x3"].size
    for name in TARGET_VARS:
        arr = data[name]
        if arr.size != nx2 * nx3:
            # allow time dimension=1 being present explicitly in dump layout flattening
            if arr.size % (nx2 * nx3) != 0:
                raise ValueError(f"Unexpected size for {name}: {arr.size}")
        data[name] = arr.reshape((-1, nx3, nx2))[0]

    return data


def split_mosaic_to_faces(field2d: np.ndarray) -> np.ndarray:
    """Convert NetCDF 3x2 cubed-sphere mosaic to faces[6, ny, nx]."""
    ny, nx = field2d.shape
    if nx % 3 != 0 or ny % 2 != 0:
        raise ValueError(f"Expected 3x2 cubed-sphere mosaic, got shape {field2d.shape}")
    nf_x = nx // 3
    nf_y = ny // 2
    if nf_x != nf_y:
        raise ValueError(f"Face tiles are not square: {nf_y} x {nf_x}")
    nf = nf_x
    tiles = field2d.reshape(2, nf, 3, nf)
    faces = np.empty((6, nf, nf), dtype=field2d.dtype)
    for r in range(2):
        for c in range(3):
            faces[r * 3 + c] = tiles[r, :, c, :]
    return faces


def extract_local_face_coords(x2: np.ndarray, x3: np.ndarray, x2f: np.ndarray, x3f: np.ndarray):
    nf = x2.size // 3
    mg = x3.size // 2
    if nf != mg:
        raise ValueError("Inconsistent face size in x2/x3")
    if x2f.size != 3 * nf + 1 or x3f.size != 2 * nf + 1:
        raise ValueError("Inconsistent mosaic edge-coordinate sizes in x2f/x3f")

    # NetCDF stores a 3x2 tiled mosaic; local face coords are the first tile ranges.
    alpha_c = x2[:nf]
    beta_c = x3[:nf]
    alpha_f = x2f[: nf + 1]
    beta_f = x3f[: nf + 1]

    return alpha_c, beta_c, alpha_f, beta_f


def face_ab_to_xyz(face: str, alpha: np.ndarray, beta: np.ndarray):
    t = np.tan(alpha)
    u = np.tan(beta)

    if face == "+X":
        X, Y, Z = np.ones_like(t), t, u
    elif face == "-X":
        X, Y, Z = -np.ones_like(t), -t, u
    elif face == "+Y":
        X, Y, Z = -t, np.ones_like(t), u
    elif face == "-Y":
        X, Y, Z = t, -np.ones_like(t), u
    elif face == "+Z":
        X, Y, Z = -u, t, np.ones_like(t)
    elif face == "-Z":
        X, Y, Z = u, t, -np.ones_like(t)
    else:
        raise ValueError(f"Invalid face {face}")

    inv = 1.0 / np.sqrt(X * X + Y * Y + Z * Z)
    return X * inv, Y * inv, Z * inv


def xyz_to_face_ab(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    ax = np.abs(x)
    ay = np.abs(y)
    az = np.abs(z)

    face = np.empty(x.shape, dtype=np.int8)
    alpha = np.empty(x.shape, dtype=np.float64)
    beta = np.empty(x.shape, dtype=np.float64)

    m = (ax >= ay) & (ax >= az) & (x >= 0)
    face[m] = 0  # +X
    t = y[m] / x[m]
    u = z[m] / x[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    m = (ax >= ay) & (ax >= az) & (x < 0)
    face[m] = 2  # -X
    t = y[m] / x[m]
    u = -z[m] / x[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    m = (ay > ax) & (ay >= az) & (y >= 0)
    face[m] = 1  # +Y
    t = -x[m] / y[m]
    u = z[m] / y[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    m = (ay > ax) & (ay >= az) & (y < 0)
    face[m] = 4  # -Y
    t = -x[m] / y[m]
    u = -z[m] / y[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    m = (az > ax) & (az > ay) & (z >= 0)
    face[m] = 3  # +Z
    t = y[m] / z[m]
    u = -x[m] / z[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    m = (az > ax) & (az > ay) & (z < 0)
    face[m] = 5  # -Z
    t = -y[m] / z[m]
    u = -x[m] / z[m]
    alpha[m] = np.arctan(t)
    beta[m] = np.arctan(u)

    return face, alpha, beta


def bilinear_sample_faces(faces: np.ndarray, alpha_f: np.ndarray, beta_f: np.ndarray, out_w=1440, out_h=720):
    lon = np.linspace(-math.pi, math.pi, out_w, endpoint=False) + math.pi / out_w
    lat = np.linspace(-math.pi / 2.0, math.pi / 2.0, out_h)  # centers incl. poles
    lon2d, lat2d = np.meshgrid(lon, lat)

    x = np.cos(lat2d) * np.cos(lon2d)
    y = np.cos(lat2d) * np.sin(lon2d)
    z = np.sin(lat2d)

    face_id, alpha, beta = xyz_to_face_ab(x, y, z)

    nx = faces.shape[-1]
    ny = faces.shape[-2]
    da = (alpha_f[-1] - alpha_f[0]) / nx
    db = (beta_f[-1] - beta_f[0]) / ny

    fa = (alpha - alpha_f[0]) / da - 0.5
    fb = (beta - beta_f[0]) / db - 0.5

    ia0 = np.floor(fa).astype(np.int64)
    ib0 = np.floor(fb).astype(np.int64)
    wa = fa - ia0
    wb = fb - ib0

    ia0 = np.clip(ia0, 0, nx - 1)
    ib0 = np.clip(ib0, 0, ny - 1)
    ia1 = np.clip(ia0 + 1, 0, nx - 1)
    ib1 = np.clip(ib0 + 1, 0, ny - 1)

    f00 = faces[face_id, ib0, ia0]
    f10 = faces[face_id, ib0, ia1]
    f01 = faces[face_id, ib1, ia0]
    f11 = faces[face_id, ib1, ia1]

    out = ((1 - wa) * (1 - wb) * f00 + wa * (1 - wb) * f10 + (1 - wa) * wb * f01 + wa * wb * f11)
    return out, np.degrees(lon2d), np.degrees(lat2d)


def percentile_range(arr: np.ndarray, lo=1.0, hi=99.0):
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    vals = arr[finite]
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colormap_viridis_like(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    # Simple 5-stop approximation (RGB in 0..1)
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([
        [0.267, 0.005, 0.329],
        [0.230, 0.322, 0.546],
        [0.128, 0.567, 0.551],
        [0.369, 0.789, 0.383],
        [0.993, 0.906, 0.144],
    ])
    rgb = np.empty(t.shape + (3,), dtype=np.float64)
    for c in range(3):
        rgb[..., c] = np.interp(t, stops, colors[:, c])
    return (255.0 * rgb).astype(np.uint8)


def add_grid_lines(rgb: np.ndarray):
    h, w, _ = rgb.shape
    # lon grid every 30 deg, lat grid every 30 deg
    for lon_deg in range(-180, 180, 30):
        x = int((lon_deg + 180.0) / 360.0 * w)
        x = np.clip(x, 0, w - 1)
        rgb[:, x : x + 1] = (rgb[:, x : x + 1] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)
    for lat_deg in range(-60, 90, 30):
        y = int(round((90.0 - lat_deg) / 180.0 * (h - 1)))
        y = np.clip(y, 0, h - 1)
        rgb[y : y + 1, :] = (rgb[y : y + 1, :] * 0.5 + np.array([255, 255, 255]) * 0.5).astype(np.uint8)


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes(order="C"))


def make_image(data2d: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    vmin, vmax = percentile_range(data2d, 1.0, 99.0)
    scaled = (data2d - vmin) / (vmax - vmin)
    rgb = colormap_viridis_like(scaled)
    nanmask = ~np.isfinite(data2d)
    if nanmask.any():
        rgb[nanmask] = np.array([0, 0, 0], dtype=np.uint8)
    add_grid_lines(rgb)
    return rgb, (vmin, vmax)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cubed-sphere path variables in spherical coordinates")
    ap.add_argument("nc_file", help="Input NetCDF file (snapy out2 .nc)")
    ap.add_argument("--outdir", default="sub_neptune/plots", help="Output directory")
    ap.add_argument("--width", type=int, default=1440, help="Output image width")
    ap.add_argument("--height", type=int, default=720, help="Output image height")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_arrays_via_ncdump(args.nc_file)
    alpha_c, beta_c, alpha_f, beta_f = extract_local_face_coords(raw["x2"], raw["x3"], raw["x2f"], raw["x3f"])

    # Emit a small metadata file documenting face ordering used.
    meta_path = outdir / "cubed_sphere_projection_metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("NetCDF cubed-sphere mosaic layout (from snapy/src/output/netcdf.cpp lines 139-147):\n")
        f.write("  row 0: faces 0(+X), 1(+Y), 2(-X)\n")
        f.write("  row 1: faces 3(+Z), 4(-Y), 5(-Z)\n")
        f.write(f"Face resolution inferred: {alpha_c.size} x {beta_c.size}\n")
        f.write(f"Local alpha range: [{alpha_f[0]}, {alpha_f[-1]}] rad\n")
        f.write(f"Local beta range:  [{beta_f[0]}, {beta_f[-1]}] rad\n")

    for var in TARGET_VARS:
        faces = split_mosaic_to_faces(raw[var])
        sphere_map, lon_deg, lat_deg = bilinear_sample_faces(
            faces, alpha_f, beta_f, out_w=args.width, out_h=args.height
        )
        rgb, (vmin, vmax) = make_image(sphere_map)
        img_path = outdir / f"{Path(args.nc_file).stem}.{var}.lonlat.ppm"
        write_ppm(img_path, rgb)

        stats_path = outdir / f"{Path(args.nc_file).stem}.{var}.stats.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            vals = raw[var]
            f.write(f"variable: {var}\n")
            f.write(f"input_shape: {vals.shape}\n")
            f.write(f"input_min: {np.nanmin(vals)}\n")
            f.write(f"input_max: {np.nanmax(vals)}\n")
            f.write(f"display_vmin_p1: {vmin}\n")
            f.write(f"display_vmax_p99: {vmax}\n")
            f.write(f"output_image: {img_path.name}\n")

        print(f"Wrote {img_path}")
        print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
