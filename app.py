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

    # ── Layers (top, no Labels column) ────────────────────────────────────────
    st.subheader("Layers")
    st.caption("Set to 0 to hide a layer.")
    col1, col2 = st.columns(2)
    with col1:
        layer_outer = int(st.number_input("Outer", 0, 255, 1))
        name_outer  = st.text_input("Outer name", "Outer", label_visibility="collapsed",
                                     placeholder="Outer layer name")
    with col2:
        layer_inner = int(st.number_input("Inner", 0, 255, 2))
        name_inner  = st.text_input("Inner name", "Inner", label_visibility="collapsed",
                                     placeholder="Inner layer name")

    st.divider()

    # ── Geometry ──────────────────────────────────────────────────────────────
    num_rows = st.slider("Rows", 1, 8, 2)
    num_cols = st.slider("Columns", 1, 8, 3)

    outer_size = st.number_input("Outer square size (μm)", 10.0, 2000.0, 100.0, 5.0)
    if layer_inner != 0:
        set_diff = st.number_input(
            "Set difference (μm)", 1.0, float(outer_size / 2 - 1), 15.0, 1.0,
            help="Inset from outer square to inner corner brackets",
        )
    else:
        set_diff = 15.0
    arm_len    = st.number_input("Corner arm length (μm)", 2.0, float(outer_size / 2), 20.0, 1.0)
    line_width = st.number_input("Line width (μm)", 0.1, 20.0, 3.0, 0.5)

    pitch_x = st.number_input("Pitch X (μm)", 10.0, 10000.0, float(outer_size * 2), 10.0)
    pitch_y = st.number_input("Pitch Y (μm)", 10.0, 10000.0, float(outer_size * 2), 10.0)

    st.divider()

    # ── Labels ────────────────────────────────────────────────────────────────
    st.subheader("Labels")
    label_layer = int(st.number_input("Label layer (0 = off)", 0, 255, 10))

    if label_layer != 0:
        label_mode = st.radio(
            "Mode",
            ["Same on every marker", "Incrementing (1, 2, 3…)"],
            horizontal=True,
        )
        if label_mode == "Same on every marker":
            label_text = st.text_input("Label text", "M")
        else:
            label_text = ""

        num_labels    = st.slider("Labels per marker", 1, 12, 1)
        label_spacing = st.number_input(
            "Spacing between labels (μm)", 1.0, 500.0, float(outer_size / 4), 1.0,
            help="Distance between consecutive labels placed around each marker",
        ) if num_labels > 1 else 0.0
        label_size    = st.number_input("Text size (μm)", 1.0, 50.0, 8.0, 1.0)
    else:
        label_mode    = "Same on every marker"
        label_text    = ""
        num_labels    = 0
        label_spacing = 0.0
        label_size    = 8.0

# ── Geometry helpers ───────────────────────────────────────────────────────────

def corner_rects(cx, cy, size, arm, lw):
    h = size / 2
    return [
        (cx - h,        cy + h - lw,  arm,       lw),
        (cx - h,        cy + h - arm, lw,        arm - lw),
        (cx + h - arm,  cy + h - lw,  arm,       lw),
        (cx + h - lw,   cy + h - arm, lw,        arm - lw),
        (cx - h,        cy - h,       arm,       lw),
        (cx - h,        cy - h + lw,  lw,        arm - lw),
        (cx + h - arm,  cy - h,       arm,       lw),
        (cx + h - lw,   cy - h + lw,  lw,        arm - lw),
    ]

def add_corners_gdspy(cell, cx, cy, size, arm, lw, layer):
    h = size / 2
    bars = [
        ((cx-h,       cy+h-lw),  (cx-h+arm, cy+h)),
        ((cx-h,       cy+h-arm), (cx-h+lw,  cy+h-lw)),
        ((cx+h-arm,   cy+h-lw),  (cx+h,     cy+h)),
        ((cx+h-lw,    cy+h-arm), (cx+h,     cy+h-lw)),
        ((cx-h,       cy-h),     (cx-h+arm, cy-h+lw)),
        ((cx-h,       cy-h+lw),  (cx-h+lw,  cy-h+arm)),
        ((cx+h-arm,   cy-h),     (cx+h,     cy-h+lw)),
        ((cx+h-lw,    cy-h+lw),  (cx+h,     cy-h+arm)),
    ]
    for p1, p2 in bars:
        cell.add(gdspy.Rectangle(p1, p2, layer=layer))


def get_labels_for_marker(marker_idx, p):
    """Return list of label strings to place for this marker."""
    if p["label_layer"] == 0 or p["num_labels"] == 0:
        return []
    n = p["num_labels"]
    if p["label_mode"] == "Same on every marker":
        return [p["label_text"] or "M"] * n
    else:
        # Incrementing: each marker gets a consecutive block
        start = marker_idx * n + 1
        return [str(start + i) for i in range(n)]


def label_positions(cx, cy, outer, n, spacing):
    """
    Return (x, y) for each of n labels, spread horizontally
    just above the top of the marker.
    """
    y = cy + outer / 2
    if n == 1:
        return [(cx, y)]
    total_width = (n - 1) * spacing
    x0 = cx - total_width / 2
    return [(x0 + i * spacing, y) for i in range(n)]

# ── GDS generation ─────────────────────────────────────────────────────────────

def generate_gds(p):
    gdspy.current_library = gdspy.GdsLibrary()
    lib  = gdspy.current_library
    cell = lib.new_cell("MARKERS")

    outer  = p["outer_size"]
    inner  = outer - 2 * p["set_diff"]
    arm    = p["arm_len"]
    lw     = p["line_width"]
    px, py = p["pitch_x"], p["pitch_y"]

    positions = []
    marker_idx = 0
    for r in range(p["num_rows"]):
        for c in range(p["num_cols"]):
            cx = c * px
            cy = -r * py
            positions.append((r, c, cx, cy))

            if p["layer_outer"] != 0:
                add_corners_gdspy(cell, cx, cy, outer, arm, lw, p["layer_outer"])
            if p["layer_inner"] != 0 and inner > arm * 2:
                add_corners_gdspy(cell, cx, cy, inner, arm, lw, p["layer_inner"])

            for lbl, (lx, ly) in zip(
                get_labels_for_marker(marker_idx, p),
                label_positions(cx, cy, outer, p["num_labels"] or 1, p["label_spacing"]),
            ):
                cell.add(gdspy.Label(
                    lbl, (lx, ly),
                    anchor="s",
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
    c_out = "#ff5555"
    c_in  = "#55ff55"
    c_lbl = "#ff4444"

    marker_idx = 0
    for _, _, cx, cy in positions:
        if p["layer_outer"] != 0:
            for x, y, w, h in corner_rects(cx, cy, outer, arm, lw):
                ax.add_patch(patches.Rectangle((x, y), w, h, fc=c_out, ec="none", zorder=2))

        if p["layer_inner"] != 0 and inner_size > arm * 2:
            for x, y, w, h in corner_rects(cx, cy, inner_size, arm, lw):
                ax.add_patch(patches.Rectangle((x, y), w, h, fc=c_in, ec="none", zorder=3))

        labels = get_labels_for_marker(marker_idx, p)
        coords = label_positions(cx, cy, outer, p["num_labels"] or 1, p["label_spacing"])
        for lbl, (lx, ly) in zip(labels, coords):
            ax.text(lx, ly + lw, lbl, color=c_lbl,
                    fontsize=max(5, p["label_size"] * 0.6),
                    ha="center", va="bottom",
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
        legend.append(patches.Patch(color=c_out, label=f"L{p['layer_outer']} – {p['name_outer']}"))
    if p["layer_inner"] != 0 and inner_size > arm * 2:
        legend.append(patches.Patch(color=c_in, label=f"L{p['layer_inner']} – {p['name_inner']}"))
    if legend:
        ax.legend(handles=legend, loc="upper right",
                  facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=8)
    ax.set_title("Marker Preview", color="white", fontsize=11, pad=8)
    return fig

# ── KLayout measurements ───────────────────────────────────────────────────────

def klayout_measurements(gds_path, p):
    layout = pya.Layout()
    layout.read(gds_path)
    dbu  = layout.dbu
    top  = layout.top_cell()
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
        {"Layer": "—", "Bounding box (μm)": f"Outer size: {outer:.2f}",       "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Inner size: {inner:.2f}",        "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Set diff: {p['set_diff']:.2f}",  "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Arm length: {p['arm_len']:.2f}", "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Pitch X: {p['pitch_x']:.2f}",   "Origin (μm)": ""},
        {"Layer": "—", "Bounding box (μm)": f"Pitch Y: {p['pitch_y']:.2f}",   "Origin (μm)": ""},
    ]
    return rows

# ── Run ────────────────────────────────────────────────────────────────────────

params = dict(
    num_rows=num_rows, num_cols=num_cols,
    outer_size=outer_size, set_diff=set_diff,
    arm_len=arm_len, line_width=line_width,
    pitch_x=pitch_x, pitch_y=pitch_y,
    layer_outer=layer_outer, name_outer=name_outer,
    layer_inner=layer_inner, name_inner=name_inner,
    label_layer=label_layer, label_mode=label_mode,
    label_text=label_text, num_labels=num_labels,
    label_spacing=label_spacing, label_size=label_size,
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
