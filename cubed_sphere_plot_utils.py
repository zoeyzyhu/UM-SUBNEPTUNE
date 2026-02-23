#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

FACE_NAMES = ["+X", "+Y", "-X", "+Z", "-Y", "-Z"]
TARGET_VARS = ["path_H2O", "path_H2O_l_", "path_H2O_l_p_"]


def load_dataset(nc_file: str):
    ds = xr.open_dataset(nc_file, engine=None)
    required = ["x2", "x2f", "x3", "x3f", *TARGET_VARS]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"Missing variables in dataset: {missing}")
    return ds


def get_data_arrays(ds, variables: Iterable[str] = TARGET_VARS):
    out: dict[str, np.ndarray] = {}
    out["x2"] = np.asarray(ds["x2"].values, dtype=np.float64)
    out["x2f"] = np.asarray(ds["x2f"].values, dtype=np.float64)
    out["x3"] = np.asarray(ds["x3"].values, dtype=np.float64)
    out["x3f"] = np.asarray(ds["x3f"].values, dtype=np.float64)
    for v in variables:
        arr = ds[v].values
        if arr.ndim == 3:  # time,x3,x2
            arr = arr[0]
        out[v] = np.asarray(arr, dtype=np.float64)
    return out


def split_mosaic_to_faces(field2d: np.ndarray) -> np.ndarray:
    ny, nx = field2d.shape
    if nx % 3 != 0 or ny % 2 != 0:
        raise ValueError(f"Expected 3x2 cubed-sphere mosaic, got shape {field2d.shape}")
    nf_x = nx // 3
    nf_y = ny // 2
    if nf_x != nf_y:
        raise ValueError(f"Face tiles are not square: {nf_y} x {nf_x}")
    nf = nf_x
    faces = np.empty((6, nf, nf), dtype=field2d.dtype)
    for face in range(6):
        r, c = divmod(face, 3)
        faces[face] = field2d[r * nf : (r + 1) * nf, c * nf : (c + 1) * nf]
    return faces


def extract_local_face_coords(x2: np.ndarray, x3: np.ndarray, x2f: np.ndarray, x3f: np.ndarray):
    nf = x2.size // 3
    if x3.size != 2 * nf:
        raise ValueError("Inconsistent face size in x2/x3")
    if x2f.size != 3 * nf + 1 or x3f.size != 2 * nf + 1:
        raise ValueError("Inconsistent mosaic edge-coordinate sizes in x2f/x3f")
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
        raise ValueError(face)
    inv = 1.0 / np.sqrt(X * X + Y * Y + Z * Z)
    return X * inv, Y * inv, Z * inv


def xyz_to_face_ab(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.empty(x.shape, dtype=np.int8)
    alpha = np.empty(x.shape, dtype=np.float64)
    beta = np.empty(x.shape, dtype=np.float64)

    m = (ax >= ay) & (ax >= az) & (x >= 0)
    face[m] = 0
    alpha[m] = np.arctan(y[m] / x[m]); beta[m] = np.arctan(z[m] / x[m])

    m = (ax >= ay) & (ax >= az) & (x < 0)
    face[m] = 2
    alpha[m] = np.arctan(y[m] / x[m]); beta[m] = np.arctan(-z[m] / x[m])

    m = (ay > ax) & (ay >= az) & (y >= 0)
    face[m] = 1
    alpha[m] = np.arctan(-x[m] / y[m]); beta[m] = np.arctan(z[m] / y[m])

    m = (ay > ax) & (ay >= az) & (y < 0)
    face[m] = 4
    alpha[m] = np.arctan(-x[m] / y[m]); beta[m] = np.arctan(-z[m] / y[m])

    m = (az > ax) & (az > ay) & (z >= 0)
    face[m] = 3
    alpha[m] = np.arctan(y[m] / z[m]); beta[m] = np.arctan(-x[m] / z[m])

    m = (az > ax) & (az > ay) & (z < 0)
    face[m] = 5
    alpha[m] = np.arctan(-y[m] / z[m]); beta[m] = np.arctan(-x[m] / z[m])

    return face, alpha, beta


def bilinear_sample_faces(faces: np.ndarray, alpha_f: np.ndarray, beta_f: np.ndarray, alpha: np.ndarray, beta: np.ndarray, face_id: np.ndarray):
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
    ia0 = np.clip(ia0, 0, nx - 1); ia1 = np.clip(ia0 + 1, 0, nx - 1)
    ib0 = np.clip(ib0, 0, ny - 1); ib1 = np.clip(ib0 + 1, 0, ny - 1)
    f00 = faces[face_id, ib0, ia0]
    f10 = faces[face_id, ib0, ia1]
    f01 = faces[face_id, ib1, ia0]
    f11 = faces[face_id, ib1, ia1]
    return ((1 - wa) * (1 - wb) * f00 + wa * (1 - wb) * f10 + (1 - wa) * wb * f01 + wa * wb * f11)


def equirectangular_sample(faces: np.ndarray, alpha_f: np.ndarray, beta_f: np.ndarray, width=1440, height=720):
    lon = np.linspace(-math.pi, math.pi, width, endpoint=False) + math.pi / width
    lat = np.linspace(-math.pi / 2, math.pi / 2, height)
    lon2d, lat2d = np.meshgrid(lon, lat)
    x = np.cos(lat2d) * np.cos(lon2d)
    y = np.cos(lat2d) * np.sin(lon2d)
    z = np.sin(lat2d)
    face_id, alpha, beta = xyz_to_face_ab(x, y, z)
    field = bilinear_sample_faces(faces, alpha_f, beta_f, alpha, beta, face_id)
    return field, np.degrees(lon2d), np.degrees(lat2d)


def orthographic_sample(
    faces: np.ndarray,
    alpha_f: np.ndarray,
    beta_f: np.ndarray,
    center_lon_deg=0.0,
    center_lat_deg=0.0,
    size=900,
):
    u = np.linspace(-1.0, 1.0, size)
    v = np.linspace(-1.0, 1.0, size)
    uu, vv = np.meshgrid(u, v)
    rr2 = uu * uu + vv * vv
    visible = rr2 <= 1.0

    lon0 = math.radians(center_lon_deg)
    lat0 = math.radians(center_lat_deg)

    # local east-north-view basis (view dir points to center_lon/lat)
    east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
    north = np.array([-math.sin(lat0) * math.cos(lon0), -math.sin(lat0) * math.sin(lon0), math.cos(lat0)])
    view = np.array([math.cos(lat0) * math.cos(lon0), math.cos(lat0) * math.sin(lon0), math.sin(lat0)])

    zc = np.zeros_like(uu)
    zc[visible] = np.sqrt(1.0 - rr2[visible])
    xloc = uu
    yloc = vv

    # world xyz = xloc*east + yloc*north + zc*view
    x = xloc * east[0] + yloc * north[0] + zc * view[0]
    y = xloc * east[1] + yloc * north[1] + zc * view[1]
    z = xloc * east[2] + yloc * north[2] + zc * view[2]

    face_id, alpha, beta = xyz_to_face_ab(x, y, z)
    field = np.full_like(uu, np.nan, dtype=np.float64)
    sampled = bilinear_sample_faces(faces, alpha_f, beta_f, alpha[visible], beta[visible], face_id[visible])
    field[visible] = sampled
    return field, visible


def percentile_range(arr: np.ndarray, lo=1.0, hi=99.0):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


def write_metadata(outdir: Path, alpha_c, beta_c, alpha_f, beta_f):
    outdir.mkdir(parents=True, exist_ok=True)
    meta = outdir / "cubed_sphere_projection_metadata.txt"
    with open(meta, "w", encoding="utf-8") as f:
        f.write("NetCDF cubed-sphere mosaic layout (from snapy/src/output/netcdf.cpp lines 139-147):\n")
        f.write("  row 0: faces 0(+X), 1(+Y), 2(-X)\n")
        f.write("  row 1: faces 3(+Z), 4(-Y), 5(-Z)\n")
        f.write(f"Face resolution inferred: {alpha_c.size} x {beta_c.size}\n")
        f.write(f"Local alpha range: [{alpha_f[0]}, {alpha_f[-1]}] rad\n")
        f.write(f"Local beta range:  [{beta_f[0]}, {beta_f[-1]}] rad\n")
