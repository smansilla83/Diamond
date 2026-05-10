import streamlit as st
import gdspy
import klayout.db as pya
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Marker Generator", layout="wide", page_icon="⬜")
st.title("Alignment Marker Generator")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    # ── Layers ────────────────────────────────────────────────────────────────
    st.subheader("Layers")
    col1, col2 = st.columns(2)
    with col1:
        use_outer   = st.checkbox("Outer", value=True)
        layer_outer = int(st.number_input("Layer#", 0, 255, 1, key="lo",
                                           label_visibility="collapsed")) if use_outer else 0
    with col2:
        use_inner   = st.checkbox("Inner", value=True)
        layer_inner = int(st.number_input("Layer#", 0, 255, 2, key="li",
                                           label_visibility="collapsed")) if use_inner else 0

    st.divider()

    # ── Geometry ──────────────────────────────────────────────────────────────
    num_rows = st.slider("Rows", 1, 8, 2)
    num_cols = st.slider("Columns", 1, 8, 3)

    outer_size = st.number_input("Outer square size (μm)", 10.0, 2000.0, 100.0, 5.0)
    if use_inner:
        set_diff = st.number_input(
            "Set difference (μm)", 1.0, float(outer_size / 2 - 1), 15.0, 1.0,
            help="Inset from outer to inner corner brackets",
        )
    else:
        set_diff = 15.0
    arm_len    = st.number_input("Corner arm length (μm)", 2.0, float(outer_size / 2), 20.0, 1.0)
    line_width = st.number_input("Line width (μm)", 0.1, 20.0, 3.0, 0.5)
    pitch_x    = st.number_input("Pitch X (μm)", 10.0, 10000.0, float(outer_size * 2), 10.0)
    pitch_y    = st.number_input("Pitch Y (μm)", 10.0, 10000.0, float(outer_size * 2), 10.0)

    st.divider()

    # ── Labels ────────────────────────────────────────────────────────────────
    st.subheader("Labels")
    label_layer = int(st.number_input("Label layer (0 = off)", 0, 255, 10))

    if label_layer != 0:
        label_type = st.radio(
            "Label type",
            ["1, 2, 3…", "A1, A2, A3…", "A, B, C…", "Custom text"],
            index=0,
        )

        prefix      = ""
        custom_text = ""
        if label_type == "A1, A2, A3…":
            prefix = st.text_input("Prefix", "A", max_chars=4)
        elif label_type == "Custom text":
            custom_text = st.text_input("Text", "MARKER")

        placement = st.radio(
            "Placement",
            ["Per marker", "Per side"],
            horizontal=True,
            help="Per marker: one label above · Per side: one label on each of the 4 sides",
        )

        label_size = st.number_input("Text size (μm)", 1.0, 50.0, 8.0, 1.0)
    else:
        label_type  = "1, 2, 3…"
        prefix      = ""
        custom_text = ""
        placement   = "Per marker"
        label_size  = 8.0

# ── Label helpers ──────────────────────────────────────────────────────────────

def _seq_letter(n):
    """0→A, 1→B, …, 25→Z, 26→AA, …"""
    result = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(ord("A") + r) + result
    return result

def label_text_for(marker_idx, side_idx, p):
    lt  = p["label_type"]
    per_side = p["placement"] == "Per side"
    idx = marker_idx * 4 + side_idx if per_side else marker_idx

    if lt == "1, 2, 3…":
        return str(idx + 1)
    if lt == "A1, A2, A3…":
        return f"{p['prefix'] or 'A'}{idx + 1}"
    if lt == "A, B, C…":
        return _seq_letter(idx)
    if lt == "Custom text":
        return p["custom_text"] or "M"
    return ""


# (x, y, ha, va, gds_anchor)
_SIDE_TOP    = lambda cx, cy, h, lw: (cx,      cy+h+lw, "center", "bottom", "s")
_SIDE_RIGHT  = lambda cx, cy, h, lw: (cx+h+lw, cy,      "left",   "center", "w")
_SIDE_BOTTOM = lambda cx, cy, h, lw: (cx,      cy-h-lw, "center", "top",    "n")
_SIDE_LEFT   = lambda cx, cy, h, lw: (cx-h-lw, cy,      "right",  "center", "e")

def label_coords(cx, cy, outer, lw, placement):
    h = outer / 2
    if placement == "Per marker":
        return [_SIDE_TOP(cx, cy, h, lw)]
    return [
        _SIDE_TOP(cx, cy, h, lw),
        _SIDE_RIGHT(cx, cy, h, lw),
        _SIDE_BOTTOM(cx, cy, h, lw),
        _SIDE_LEFT(cx, cy, h, lw),
    ]

# ── Geometry helpers ───────────────────────────────────────────────────────────

def corner_rects(cx, cy, size, arm, lw):
    h = size / 2
    return [
        (cx-h,       cy+h-lw,  arm,      lw),
        (cx-h,       cy+h-arm, lw,       arm-lw),
        (cx+h-arm,   cy+h-lw,  arm,      lw),
        (cx+h-lw,    cy+h-arm, lw,       arm-lw),
        (cx-h,       cy-h,     arm,      lw),
        (cx-h,       cy-h+lw,  lw,       arm-lw),
        (cx+h-arm,   cy-h,     arm,      lw),
        (cx+h-lw,    cy-h+lw,  lw,       arm-lw),
    ]

def add_corners_gdspy(cell, cx, cy, size, arm, lw, layer):
    h = size / 2
    for p1, p2 in [
        ((cx-h,     cy+h-lw),  (cx-h+arm, cy+h)),
        ((cx-h,     cy+h-arm), (cx-h+lw,  cy+h-lw)),
        ((cx+h-arm, cy+h-lw),  (cx+h,     cy+h)),
        ((cx+h-lw,  cy+h-arm), (cx+h,     cy+h-lw)),
        ((cx-h,     cy-h),     (cx-h+arm, cy-h+lw)),
        ((cx-h,     cy-h+lw),  (cx-h+lw,  cy-h+arm)),
        ((cx+h-arm, cy-h),     (cx+h,     cy-h+lw)),
        ((cx+h-lw,  cy-h+lw),  (cx+h,     cy-h+arm)),
    ]:
        cell.add(gdspy.Rectangle(p1, p2, layer=layer))

# ── GDS generation ─────────────────────────────────────────────────────────────

def generate_gds(p):
    gdspy.current_library = gdspy.GdsLibrary()
    lib  = gdspy.current_library
    cell = lib.new_cell("MARKERS")

    outer = p["outer_size"]
    inner = outer - 2 * p["set_diff"]
    arm   = p["arm_len"]
    lw    = p["line_width"]

    positions  = []
    marker_idx = 0
    for r in range(p["num_rows"]):
        for c in range(p["num_cols"]):
            cx = c * p["pitch_x"]
            cy = -r * p["pitch_y"]
            positions.append((cx, cy))

            if p["layer_outer"] != 0:
                add_corners_gdspy(cell, cx, cy, outer, arm, lw, p["layer_outer"])
            if p["layer_inner"] != 0 and inner > arm * 2:
                add_corners_gdspy(cell, cx, cy, inner, arm, lw, p["layer_inner"])

            if p["label_layer"] != 0:
                for side_idx, (lx, ly, _, _, anchor) in enumerate(
                    label_coords(cx, cy, outer, lw, p["placement"])
                ):
                    txt = label_text_for(marker_idx, side_idx, p)
                    cell.add(gdspy.Label(
                        txt, (lx, ly), anchor=anchor,
                        magnification=p["label_size"] / 4.0,
                        layer=p["label_layer"],
                    ))
            marker_idx += 1

    return lib, cell, positions, inner

# ── Preview ────────────────────────────────────────────────────────────────────

def render(p, positions, inner_size):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")

    outer = p["outer_size"]
    arm   = p["arm_len"]
    lw    = p["line_width"]

    marker_idx = 0
    for cx, cy in positions:
        if p["layer_outer"] != 0:
            for x, y, w, h in corner_rects(cx, cy, outer, arm, lw):
                ax.add_patch(patches.Rectangle((x, y), w, h, fc="#ff5555", ec="none", zorder=2))

        if p["layer_inner"] != 0 and inner_size > arm * 2:
            for x, y, w, h in corner_rects(cx, cy, inner_size, arm, lw):
                ax.add_patch(patches.Rectangle((x, y), w, h, fc="#55ff55", ec="none", zorder=3))

        if p["label_layer"] != 0:
            for side_idx, (lx, ly, ha, va, _) in enumerate(
                label_coords(cx, cy, outer, lw, p["placement"])
            ):
                txt = label_text_for(marker_idx, side_idx, p)
                ax.text(lx, ly, txt, color="#ff4444",
                        fontsize=max(5, p["label_size"] * 0.6),
                        ha=ha, va=va,
                        fontfamily="monospace", fontweight="bold", zorder=4)
        marker_idx += 1

    ax.set_aspect("equal")
    ax.autoscale_view()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    pad = outer * 0.6
    ax.set_xlim(xl[0] - pad, xl[1] + pad)
    ax.set_ylim(yl[0] - pad, yl[1] + pad)
    ax.set_xlabel("X (μm)", color="#aaaacc", fontsize=9)
    ax.set_ylabel("Y (μm)", color="#aaaacc", fontsize=9)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444466")

    legend = []
    if p["layer_outer"] != 0:
        legend.append(patches.Patch(color="#ff5555", label=f"Layer {p['layer_outer']} – outer"))
    if p["layer_inner"] != 0 and inner_size > arm * 2:
        legend.append(patches.Patch(color="#55ff55", label=f"Layer {p['layer_inner']} – inner"))
    if legend:
        ax.legend(handles=legend, loc="upper right",
                  facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=8)
    ax.set_title("Marker Preview", color="white", fontsize=11, pad=8)
    return fig

# ── KLayout measurements ───────────────────────────────────────────────────────

def klayout_measurements(gds_path, p):
    layout = pya.Layout()
    layout.read(gds_path)
    dbu = layout.dbu
    top = layout.top_cell()
    rows = []

    for idx in range(layout.layers()):
        info = layout.get_info(idx)
        li   = layout.find_layer(info)
        if li is None:
            continue
        bbox = top.bbox_per_layer(li)
        if bbox.empty():
            continue
        rows.append({
            "Layer": f"{info.layer}/{info.datatype}",
            "Bounding box (μm)": f"{bbox.width()*dbu:.2f} × {bbox.height()*dbu:.2f}",
            "Origin (μm)": f"({bbox.left*dbu:.2f}, {bbox.bottom*dbu:.2f})",
        })

    outer = p["outer_size"]
    inner = max(0.0, outer - 2 * p["set_diff"])
    rows += [
        {"Layer": "—", "Bounding box (μm)": f"Outer size: {outer:.2f}",        "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Inner size: {inner:.2f}",         "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Set diff: {p['set_diff']:.2f}",   "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Arm length: {p['arm_len']:.2f}",  "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Pitch X: {p['pitch_x']:.2f}",     "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Pitch Y: {p['pitch_y']:.2f}",     "Origin (μm)": ""},
    ]
    return rows

# ── Run ────────────────────────────────────────────────────────────────────────

params = dict(
    num_rows=num_rows, num_cols=num_cols,
    outer_size=outer_size, set_diff=set_diff,
    arm_len=arm_len, line_width=line_width,
    pitch_x=pitch_x, pitch_y=pitch_y,
    layer_outer=layer_outer, layer_inner=layer_inner,
    label_layer=label_layer, label_type=label_type,
    prefix=prefix, custom_text=custom_text,
    placement=placement, label_size=label_size,
)

lib, cell, positions, inner_size = generate_gds(params)

tmp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
tmp.close()
lib.write_gds(tmp.name)
with open(tmp.name, "rb") as f:
    gds_bytes = f.read()

tab1, tab2 = st.tabs(["Preview", "Measurements"])

with tab1:
    fig = render(params, positions, inner_size)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab2:
    try:
        df = pd.DataFrame(klayout_measurements(tmp.name, params))
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

os.unlink(tmp.name)

st.download_button("⬇ Download GDS", gds_bytes, "markers.gds",
                   mime="application/octet-stream", type="primary")
