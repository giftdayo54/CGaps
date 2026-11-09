"""Imagery rendering and a dependency-free swipe (before/after) comparison component."""
from __future__ import annotations

import base64
import html
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio import features  # noqa: E402

from .areas import to_crs_geom, utm_crs_for  # noqa: E402
from .pipeline import indices  # noqa: E402

LAYERS = ["True colour", "False colour (NIR-Red-Green)", "NDVI", "NDRE", "NDMI", "Sentinel-1 VH (dB)", "Sentinel-1 VV (dB)"]
LEGENDS = {
    "True colour": "B04/B03/B02 stretched 0-0.25 reflectance.",
    "False colour (NIR-Red-Green)": "B08/B04/B03; healthy vegetation = red/magenta, bare soil = cyan/grey.",
    "NDVI": "Red-yellow-green colour ramp, 0 to 0.9 (grey = no valid observation).",
    "NDRE": "Colour ramp 0 to 0.6 (red-edge based canopy vigour).",
    "NDMI": "Colour ramp -0.2 to 0.6 (canopy/soil moisture indicator).",
    "Sentinel-1 VH (dB)": "Grey ramp -26 to -12 dB; depends on moisture, roughness and viewing geometry.",
    "Sentinel-1 VV (dB)": "Grey ramp -18 to -4 dB; depends on moisture, roughness and viewing geometry.",
}


def _stretch(a, lo, hi, gamma=1.0):
    return np.clip((a - lo) / (hi - lo), 0, 1) ** gamma


def render_layer(comp, layer: str, field_geom_wgs, gaps_wgs=None, comp_s1=None, pad_px: int = 3):
    """RGBA uint8 array cropped to the field's bounding box; outside the field is transparent, no-data is grey."""
    src = comp
    if layer.startswith("Sentinel-1"):
        if comp_s1 is None:
            raise ValueError("Sentinel-1 data not supplied for this layer.")
        src = comp_s1
    h, w = next(iter(src.bands.values())).shape
    g = to_crs_geom(field_geom_wgs, "EPSG:4326", src.crs)
    inside = features.geometry_mask([g], (h, w), src.transform, invert=True)
    b = comp.bands
    nodata = np.zeros((h, w), bool)
    if layer == "True colour":
        rgb = np.stack([_stretch(b["B04"], 0, 0.25, 0.8), _stretch(b["B03"], 0, 0.25, 0.8), _stretch(b["B02"], 0, 0.25, 0.8)], -1)
    elif layer.startswith("False colour"):
        rgb = np.stack([_stretch(b["B08"], 0, 0.5), _stretch(b["B04"], 0, 0.25), _stretch(b["B03"], 0, 0.25)], -1)
    elif layer in ("NDVI", "NDRE", "NDMI"):
        ix = indices(b)[layer]
        lo, hi = {"NDVI": (0.0, 0.9), "NDRE": (0.0, 0.6), "NDMI": (-0.2, 0.6)}[layer]
        rgb = plt.get_cmap("RdYlGn")(_stretch(np.nan_to_num(ix, nan=lo), lo, hi))[..., :3]
        nodata = ~np.isfinite(ix)
    else:
        key = "VH" if "VH" in layer else "VV"
        lin = src.bands[key]
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10 * np.log10(np.where(lin > 0, lin, np.nan))
        lo, hi = (-26, -12) if key == "VH" else (-18, -4)
        v = _stretch(np.nan_to_num(db, nan=lo), lo, hi)
        rgb = np.stack([v, v, v], -1)
        nodata = ~np.isfinite(db)
    if layer in ("True colour",) or layer.startswith("False colour"):
        nodata = ~np.isfinite(np.stack([b["B04"], b["B03"]], -1)).all(-1)
        rgb = np.nan_to_num(rgb, nan=0.0)
    img = np.zeros((h, w, 4), dtype="uint8")
    img[..., :3] = (rgb * 255).astype("uint8")
    img[nodata, :3] = 190
    img[..., 3] = np.where(inside, 255, 0)
    if gaps_wgs:
        gu = [to_crs_geom(x, "EPSG:4326", src.crs) for x in gaps_wgs]
        edge = features.rasterize([(x.boundary.buffer(src.resolution_m * 0.6), 1) for x in gu], (h, w),
                                  transform=src.transform, fill=0, dtype="uint8").astype(bool) & inside
        img[edge] = (213, 0, 0, 255)
    ys, xs = np.where(inside)
    r0, r1, c0, c1 = max(ys.min() - pad_px, 0), min(ys.max() + pad_px + 1, h), max(xs.min() - pad_px, 0), min(xs.max() + pad_px + 1, w)
    return img[r0:r1, c0:c1]


def to_data_uri(rgba: np.ndarray, scale: int = 1) -> str:
    im = Image.fromarray(rgba, "RGBA")
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def swipe_html(uri_a: str, uri_b: str, label_a: str, label_b: str, aspect: float, max_width: int = 760) -> str:
    """Self-contained before/after slider (image A under, image B revealed from the right). No external JS."""
    la, lb = html.escape(label_a), html.escape(label_b)
    h = int(max_width / max(aspect, 0.2)) + 70
    return f"""
<div style="font-family:sans-serif;max-width:{max_width}px">
  <div id="cg-wrap" style="position:relative;width:100%;aspect-ratio:{aspect:.4f};background:#eceff1;border:1px solid #bbb;overflow:hidden">
    <img src="{uri_a}" style="position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated">
    <img id="cg-top" src="{uri_b}" style="position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated;clip-path:inset(0 0 0 50%)">
    <div id="cg-bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 3px #000"></div>
    <div style="position:absolute;top:6px;left:6px;background:#000a;color:#fff;padding:2px 6px;font-size:12px">{la}</div>
    <div style="position:absolute;top:6px;right:6px;background:#000a;color:#fff;padding:2px 6px;font-size:12px">{lb}</div>
  </div>
  <input id="cg-range" type="range" min="0" max="100" value="50" style="width:100%">
  <script>
    const r=document.getElementById('cg-range'),t=document.getElementById('cg-top'),b=document.getElementById('cg-bar');
    r.addEventListener('input',()=>{{t.style.clipPath='inset(0 0 0 '+r.value+'%)';b.style.left=r.value+'%';}});
  </script>
</div>""", h


def render_pair(provider, field_geom_wgs, field_id, layer, range_a, range_b, res=10.0, gaps_wgs=None, orbit=None):
    """Fetch two composites for one field and render the chosen layer for each. Raises provider errors."""
    utm = utm_crs_for(field_geom_wgs)
    g = to_crs_geom(field_geom_wgs, "EPSG:4326", utm)
    out = {}
    for tag, (d0, d1) in (("a", range_a), ("b", range_b)):
        comp = provider.get_composite(g.bounds, str(utm), d0, d1, res, seed_key=field_id)
        s1 = provider.get_s1_composite(g.bounds, str(utm), d0, d1, res, seed_key=field_id) if layer.startswith("Sentinel-1") else None
        rgba = render_layer(comp, layer, field_geom_wgs, gaps_wgs if tag == "b" else None, s1)
        finite = np.isfinite(comp.bands["B04"])
        h, w = finite.shape
        inside = features.geometry_mask([g], (h, w), comp.transform, invert=True)
        out[tag] = {"rgba": rgba, "comp": comp, "s1": s1, "valid_fraction": float((finite & inside).sum() / max(inside.sum(), 1)),
                    "range": (str(d0), str(d1)), "scenes": comp.scenes + (s1.scenes if s1 else []), "synthetic": comp.synthetic}
    return out
