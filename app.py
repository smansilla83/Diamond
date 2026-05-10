import streamlit as st
import gdspy
import klayout.db as pya
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import os
import pandas as pd

st.set_page_config(page_title="Alignment Marker Generator", layout="wide", page_icon="⬜")
st.title("Alignment Marker Generator")
st.caption(
    "Generate GDS alignment markers with **gdspy** · "
    "Measure with **KLayout** · Preview with Matplotlib"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Marker Parameters")

    num_markers = st.slider("Number of markers", 1, 16, 4)
    outer_size = st.number_input(
        "Outer square size (μm)", 10.0, 2000.0, 100.0, 5.0,
        help="Side length of the outer square frame",
    )
    set_diff = st.number_input(
        "Set difference – gap (μm)", 1.0, 500.0, 10.0, 1.0,
        help="Gap between outer frame inner-edge and inner frame outer-edge",
    )
    line_width = st.number_input(
        "Line width (μm)", 0.1, 20.0, 2.0, 0.5,
        help="Width of each frame bar",
    )
    marker_pitch = st.number_input(
        "Marker pitch (μm)", 10.0, 10000.0, float(outer_size * 1.8), 10.0,
        help="Center-to-center distance between markers",
    )

    if marker_pitch < outer_size:
        st.warning("Pitch is smaller than outer size – markers will overlap.")

    st.divider()
    arrangement = st.radio("Arrangement", ["Row (horizontal)", "Column (vertical)", "Grid"])

    st.divider()
    st.subheader("Labels")
    label_type = st.selectbox(
        "Label type",
        ["None", "Index (M1, M2 …)", "Coordinates (x, y)", "Set-difference value", "Custom text"],
    )
    custom_text = ""
    label_pos  = "Top"
    label_size = 5.0
    if label_type != "None":
        label_pos  = st.selectbox("Label position", ["Top", "Bottom", "Left", "Right"])
        label_size = st.number_input("Text height (μm)", 0.5, 100.0, 5.0, 0.5)
        if label_type == "Custom text":
            custom_text = st.text_input("Custom text", "MARKER")

    st.divider()
    st.subheader("Layers")
    col1, col2 = st.columns(2)
    with col1:
        layer_outer = int(st.number_input("Outer", 0, 255, 1, key="l_outer"))
        layer_inner = int(st.number_input("Inner", 0, 255, 2, key="l_inner"))
    with col2:
        layer_label = int(st.number_input("Labels", 0, 255, 10, key="l_label"))

    st.divider()
    show_dims = st.toggle("Show dimension annotations", True)
    show_grid  = st.toggle("Show grid", False)

# ── Geometry helpers ───────────────────────────────────────────────────────────

def _frame_bars_gdspy(cx, cy, size, lw):
    h = size / 2.0
    return [
        ((cx - h,      cy + h - lw), (cx + h,      cy + h)),
        ((cx - h,      cy - h),      (cx + h,      cy - h + lw)),
        ((cx - h,      cy - h + lw), (cx - h + lw, cy + h - lw)),
        ((cx + h - lw, cy - h + lw), (cx + h,      cy + h - lw)),
    ]

def _frame_bars_mpl(cx, cy, size, lw):
    """(x, y, w, h) for each bar of the hollow square."""
    h = size / 2.0
    return [
        (cx - h,      cy + h - lw,  size,        lw),
        (cx - h,      cy - h,       size,        lw),
        (cx - h,      cy - h + lw,  lw,          size - 2 * lw),
        (cx + h - lw, cy - h + lw,  lw,          size - 2 * lw),
    ]

def make_hollow_square_gdspy(cell, cx, cy, size, lw, layer):
    for pt1, pt2 in _frame_bars_gdspy(cx, cy, size, lw):
        cell.add(gdspy.Rectangle(pt1, pt2, layer=layer))

def compute_positions(num, pitch, arrangement):
    if arrangement.startswith("Row"):
        return [(i * pitch, 0.0) for i in range(num)]
    elif arrangement.startswith("Col"):
        return [(0.0, -i * pitch) for i in range(num)]
    else:
        cols = int(np.ceil(np.sqrt(num)))
        return [(i % cols * pitch, -(i // cols) * pitch) for i in range(num)]

def label_xy(cx, cy, outer, lsize, pos):
    margin = outer / 2.0 + lsize * 2.5
    if pos == "Top":     return cx,         cy + margin
    if pos == "Bottom":  return cx,         cy - margin
    if pos == "Left":    return cx - margin, cy
    return                      cx + margin, cy

GDSPY_ANCHOR = {"Top": "s", "Bottom": "n", "Left": "e", "Right": "w"}

# ── GDS generation ─────────────────────────────────────────────────────────────

def generate_gds(p):
    lib  = gdspy.GdsLibrary()
    cell = lib.new_cell("MARKERS")

    outer      = p["outer_size"]
    lw         = p["line_width"]
    diff       = p["set_diff"]
    inner_size = max(0.0, outer - 4 * lw - 2 * diff)
    positions  = compute_positions(p["num_markers"], p["marker_pitch"], p["arrangement"])

    for i, (cx, cy) in enumerate(positions):
        make_hollow_square_gdspy(cell, cx, cy, outer, lw, p["layer_outer"])
        if inner_size > lw * 2:
            make_hollow_square_gdspy(cell, cx, cy, inner_size, lw, p["layer_inner"])

        if p["label_type"] != "None":
            text = _resolve_label(p, i, cx, cy)
            lx, ly = label_xy(cx, cy, outer, p["label_size"], p["label_pos"])
            cell.add(gdspy.Label(
                text, (lx, ly),
                anchor=GDSPY_ANCHOR[p["label_pos"]],
                magnification=p["label_size"] / 4.0,
                layer=p["layer_label"],
            ))

    return lib, cell, positions, inner_size

def _resolve_label(p, idx, cx, cy):
    lt = p["label_type"]
    if lt == "Index (M1, M2 …)":       return f"M{idx + 1}"
    if lt == "Coordinates (x, y)":     return f"({cx:.0f}, {cy:.0f})"
    if lt == "Set-difference value":   return f"d={p['set_diff']} μm"
    if lt == "Custom text":            return p["custom_text"] or "MARKER"
    return f"M{idx + 1}"

# ── Matplotlib preview ─────────────────────────────────────────────────────────

_LAYER_PALETTE = [
    "#e63946", "#2a9d8f", "#e9c46a", "#457b9d",
    "#a8dadc", "#f4a261", "#e76f51", "#06d6a0",
]

def _layer_color(lnum, lo, li, ll):
    fixed = {lo: "#ff5555", li: "#55ff55", ll: "#ffff55"}
    return fixed.get(lnum, _LAYER_PALETTE[lnum % len(_LAYER_PALETTE)])

def _draw_dim(ax, x1, y1, x2, y2, label, color="#999999", fs=7):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.9),
    )
    ax.text(
        (x1 + x2) / 2, (y1 + y2) / 2, label,
        color=color, fontsize=fs, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="none", alpha=0.85),
    )

def render_preview(p, positions, inner_size):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")

    lo, li, ll = p["layer_outer"], p["layer_inner"], p["layer_label"]
    c_out = _layer_color(lo, lo, li, ll)
    c_in  = _layer_color(li, lo, li, ll)
    c_lbl = _layer_color(ll, lo, li, ll)

    outer = p["outer_size"]
    lw    = p["line_width"]
    diff  = p["set_diff"]

    for i, (cx, cy) in enumerate(positions):
        for x, y, w, h in _frame_bars_mpl(cx, cy, outer, lw):
            ax.add_patch(patches.Rectangle((x, y), w, h, fc=c_out, ec="none", alpha=0.85, zorder=2))

        if inner_size > lw * 2:
            for x, y, w, h in _frame_bars_mpl(cx, cy, inner_size, lw):
                ax.add_patch(patches.Rectangle((x, y), w, h, fc=c_in, ec="none", alpha=0.85, zorder=3))

        if p["label_type"] != "None":
            text = _resolve_label(p, i, cx, cy)
            lx, ly = label_xy(cx, cy, outer, p["label_size"], p["label_pos"])
            ax.text(lx, ly, text, color=c_lbl,
                    fontsize=max(5, p["label_size"] * 0.55),
                    ha="center", va="center", zorder=4, fontfamily="monospace")

    # Dimension annotations on the first marker
    if p["show_dims"] and positions:
        cx, cy = positions[0]
        ho = outer / 2
        hi = inner_size / 2

        pad = outer * 0.18
        # Outer width
        _draw_dim(ax, cx - ho, cy - ho - pad, cx + ho, cy - ho - pad, f"{outer:.1f} μm")
        # Outer height
        _draw_dim(ax, cx - ho - pad, cy - ho, cx - ho - pad, cy + ho, f"{outer:.1f} μm")

        if inner_size > lw * 2:
            # Gap (set difference) on the right side
            _draw_dim(ax, cx + hi + lw, cy, cx + ho - lw, cy, f"Δ = {diff:.1f} μm")
            # Inner size
            _draw_dim(ax, cx - hi, cy + hi + pad * 0.6, cx + hi, cy + hi + pad * 0.6,
                      f"{inner_size:.1f} μm")

    if p["show_grid"]:
        ax.grid(True, color="#2a2a55", lw=0.4, alpha=0.6)

    ax.set_aspect("equal")
    ax.autoscale_view()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    pad = outer * 0.5
    ax.set_xlim(xl[0] - pad, xl[1] + pad)
    ax.set_ylim(yl[0] - pad, yl[1] + pad)

    ax.set_xlabel("X (μm)", color="#aaaacc", fontsize=9)
    ax.set_ylabel("Y (μm)", color="#aaaacc", fontsize=9)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444466")

    legend = [
        patches.Patch(color=c_out, label=f"Layer {lo} – outer"),
        patches.Patch(color=c_in,  label=f"Layer {li} – inner"),
    ]
    if p["label_type"] != "None":
        legend.append(patches.Patch(color=c_lbl, label=f"Layer {ll} – labels"))
    ax.legend(handles=legend, loc="upper right",
              facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=8)
    ax.set_title("Marker Preview (matplotlib · KLayout-style)", color="white", fontsize=11, pad=10)

    return fig

# ── KLayout measurements ───────────────────────────────────────────────────────

def klayout_measurements(gds_path, p):
    layout = pya.Layout()
    layout.read(gds_path)
    dbu  = layout.dbu
    top  = layout.top_cell()
    rows = []

    # Bounding boxes per layer (from klayout.db)
    for idx in range(layout.layers()):
        info = layout.get_info(idx)
        li   = layout.find_layer(info)
        if li is None:
            continue
        bbox = top.bbox_per_layer(li)
        if bbox.empty():
            continue
        rows.append({
            "Source": "klayout.db",
            "Measurement": f"Layer {info.layer}/{info.datatype} bounding box",
            "Value (μm)": f"{bbox.width() * dbu:.3f} × {bbox.height() * dbu:.3f}",
            "Details": (
                f"({bbox.left * dbu:.3f}, {bbox.bottom * dbu:.3f}) → "
                f"({bbox.right * dbu:.3f}, {bbox.top * dbu:.3f})"
            ),
        })

    # Shape count per layer
    for idx in range(layout.layers()):
        info = layout.get_info(idx)
        li   = layout.find_layer(info)
        if li is None:
            continue
        count = top.shapes(li).size()
        if count == 0:
            continue
        rows.append({
            "Source": "klayout.db",
            "Measurement": f"Layer {info.layer}/{info.datatype} shape count",
            "Value (μm)": str(count),
            "Details": "Number of polygon/rect objects",
        })

    # Derived analytical measurements
    outer      = p["outer_size"]
    lw         = p["line_width"]
    diff       = p["set_diff"]
    inner_size = max(0.0, outer - 4 * lw - 2 * diff)
    gap_actual = (outer - inner_size) / 2 - lw  # gap between bar edges

    derived = [
        ("Outer square size",          f"{outer:.3f}",                  "Side length of outer frame"),
        ("Outer frame line width",      f"{lw:.3f}",                     "Width of each bar"),
        ("Set difference (gap)",        f"{diff:.3f}",                   "Space from outer inner-edge to inner outer-edge"),
        ("Inner square size",           f"{inner_size:.3f}",             f"outer − 4·lw − 2·gap = {outer} − {4*lw} − {2*diff}"),
        ("Actual inter-frame gap",      f"{gap_actual:.3f}",             "From inner frame outer-edge to outer frame inner-edge"),
        ("Outer inner clearance",       f"{outer / 2 - lw:.3f}",        "Half outer size minus line width"),
        ("Marker pitch",                f"{p['marker_pitch']:.3f}",      "Center-to-center distance between markers"),
        ("Number of markers",           str(p["num_markers"]),           ""),
    ]
    if p["num_markers"] > 1:
        span = p["marker_pitch"] * (p["num_markers"] - 1)
        derived.append(("Total array span", f"{span:.3f}", "First to last marker center"))

    for name, val, details in derived:
        rows.append({"Source": "analytical", "Measurement": name,
                     "Value (μm)": val, "Details": details})

    return rows

# ── Assemble params dict ───────────────────────────────────────────────────────

params = dict(
    num_markers=num_markers,
    outer_size=outer_size,
    set_diff=set_diff,
    line_width=line_width,
    marker_pitch=marker_pitch,
    arrangement=arrangement,
    label_type=label_type,
    label_pos=label_pos,
    label_size=label_size,
    custom_text=custom_text,
    layer_outer=layer_outer,
    layer_inner=layer_inner,
    layer_label=layer_label,
    show_dims=show_dims,
    show_grid=show_grid,
)

# ── Generate ───────────────────────────────────────────────────────────────────

lib, cell, positions, inner_size = generate_gds(params)

tmp = tempfile.NamedTemporaryFile(suffix=".gds", delete=False)
tmp.close()
lib.write_gds(tmp.name)

with open(tmp.name, "rb") as f:
    gds_bytes = f.read()

# ── Layout ────────────────────────────────────────────────────────────────────

tab_preview, tab_measure = st.tabs(["Preview", "Measurements (KLayout)"])

with tab_preview:
    fig = render_preview(params, positions, inner_size)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if inner_size <= line_width * 2:
        st.warning(
            f"Inner square size ({inner_size:.2f} μm) is too small for the current line width. "
            "Reduce the set difference or line width, or increase the outer square size."
        )

with tab_measure:
    st.subheader("Distances & Bounding Boxes")
    st.caption("Bounding boxes read from the GDS file via **klayout.db**; remaining rows are analytical.")
    try:
        rows = klayout_measurements(tmp.name, params)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Source":      st.column_config.TextColumn(width="small"),
                         "Measurement": st.column_config.TextColumn(width="medium"),
                         "Value (μm)":  st.column_config.TextColumn(width="small"),
                         "Details":     st.column_config.TextColumn(width="large"),
                     })
    except Exception as e:
        st.error(f"KLayout measurement error: {e}")

st.divider()
col_dl, col_info = st.columns([1, 3])
with col_dl:
    st.download_button(
        "⬇ Download GDS",
        data=gds_bytes,
        file_name="markers.gds",
        mime="application/octet-stream",
        type="primary",
    )
with col_info:
    st.info(
        "Open the downloaded **markers.gds** in KLayout desktop for full interactive visualization, "
        "layer color editing, and ruler measurements."
    )

os.unlink(tmp.name)
