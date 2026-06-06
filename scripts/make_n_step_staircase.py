#!/usr/bin/env python3
"""
Generate an N-box staircase artifact that extends the original 3-box
`walk_up_karen_stairs` staircase (Karen's retargeted stairs) to an arbitrary
number of steps, writing it into a self-contained artifact dir with the SAME
layout the Staircase task expects:

    <out_dir>/
        multi_boxes.urdf            # N box links (collision) -> box_models/box{i}.obj
        multi_boxes.obj             # single combined mesh (used by the raycast height scanner)
        multi_boxes_combined.urdf   # wraps multi_boxes.obj
        box_models/box{1..N}.obj    # per-step solid boxes
        staircase_metadata.json     # bounds for record-keeping

Geometry model (derived exactly from the original 3-box artifact):
  - Each box is a vertical prism from z=0 up to its top height.
  - Footprint is a slightly-skewed quad propagated by a constant per-step
    "run" vector R; consecutive boxes share an edge (front edge of box i ==
    back edge of box i-1), so the stack is watertight and regular.
  - Boxes 1..N-1 are normal risers; box N is the elongated top LANDING
    (run extended LANDING_RUN_SCALE x, like the original box3).
  - Rise is uniform; step 1 gets a small +STEP1_OFFSET bump (matches original).

The climb (local -x) direction reproduces the original artifact EXACTLY; the
small (~1-4 cm) per-box width-registration noise in the hand-made original is
intentionally dropped in favor of a clean regular staircase.

Usage:
    python scripts/make_n_step_staircase.py --num_boxes 6 \
        --out_dir /move/u/chrzhang/rmr_tracking/artifacts/walk_up_karen_stairs_6step
"""
import argparse
import json
import os

import numpy as np

# --- Geometry constants, derived from the original walk_up_karen_stairs boxes ---
R0 = np.array([0.0, 0.28322201])          # step-1 front-right corner (y+ rail), z=0
L0 = np.array([-0.05028353, -1.06033911])  # step-1 front-left  corner (y- rail), z=0
RUN = np.array([-0.33074625, 0.03865124])  # per-step run vector (local -x climb dir)
RISE = 0.17558225                          # uniform per-step rise (m)
STEP1_OFFSET = 0.02                        # extra height bump on step 1 (matches original)
LANDING_RUN_SCALE = 3.0                    # top landing is this many runs deep (orig box3)

# Face template for an 8-vertex box: bottom verts 1-4 (z=0), top verts 5-8
# (same x,y extruded). Copied verbatim from the original box{1,2,3}.obj so the
# winding/normals match exactly.
_FACES = [
    (1, 3, 2), (1, 4, 3),          # bottom
    (5, 6, 7), (5, 7, 8),          # top
    (1, 2, 6), (1, 6, 5),          # sides
    (4, 8, 7), (4, 7, 3),
    (1, 5, 8), (1, 8, 4),
    (2, 3, 7), (2, 7, 6),
]

_COLORS = [
    "0.3 0.7 0.9 0.5", "0.7 0.3 0.9 0.5", "0.9 0.7 0.3 0.5",
    "0.3 0.9 0.7 0.5", "0.9 0.3 0.7 0.5", "0.7 0.9 0.3 0.5",
]


def _rail(point0, k):
    """Base (z=0) xy position of a rail corner after k runs."""
    return point0 + k * RUN


def box_corners(i, num_boxes):
    """Return (bottom4 xy [front-right, front-left, back-left, back-right], top_z)
    for box index i (1-based). Box num_boxes is the elongated landing."""
    front_r = _rail(R0, i - 1)
    front_l = _rail(L0, i - 1)
    if i == num_boxes:
        # Elongated top landing.
        back_l = front_l + LANDING_RUN_SCALE * RUN
        back_r = front_r + LANDING_RUN_SCALE * RUN
    else:
        back_l = _rail(L0, i)
        back_r = _rail(R0, i)
    top_z = RISE * i + (STEP1_OFFSET if i == 1 else 0.0)
    # order matches original box obj: front-right, front-left, back-left, back-right
    return np.array([front_r, front_l, back_l, back_r]), top_z


def box_vertices(i, num_boxes):
    """8 (x,y,z) vertices for box i: bottom 4 (z=0) then top 4 (z=top)."""
    quad_xy, top_z = box_corners(i, num_boxes)
    bottom = np.column_stack([quad_xy, np.zeros(4)])
    top = np.column_stack([quad_xy, np.full(4, top_z)])
    return np.vstack([bottom, top]), top_z


def write_box_obj(path, name, verts):
    lines = [f"# {name}"]
    for v in verts:
        lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
    for f in _FACES:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_combined_obj(path, name, all_verts):
    """all_verts: list of (8,3) arrays per box. Faces offset by 8 per box."""
    lines = [f"# {name}"]
    for verts in all_verts:
        for v in verts:
            lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
    for bi in range(len(all_verts)):
        off = 8 * bi
        for f in _FACES:
            lines.append(f"f {f[0] + off} {f[1] + off} {f[2] + off}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_multi_boxes_urdf(path, num_boxes):
    parts = ['<?xml version="1.0"?>', '<robot name="multi_boxes">', '  <link name="world"/>', ""]
    for i in range(1, num_boxes + 1):
        color = _COLORS[(i - 1) % len(_COLORS)]
        parts += [
            f'  <link name="multi_boxes_box{i}_link">',
            '    <visual>',
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            '      <geometry>',
            f'        <mesh filename="box_models/box{i}.obj" scale="1 1 1"/>',
            '      </geometry>',
            f'      <material name="box{i}_material">',
            f'        <color rgba="{color}"/>',
            '      </material>',
            '    </visual>',
            '    <collision>',
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            '      <geometry>',
            f'        <mesh filename="box_models/box{i}.obj" scale="1 1 1"/>',
            '      </geometry>',
            '    </collision>',
            '    <inertial>',
            '      <mass value="33.33"/>',
            '      <origin rpy="0 0 0" xyz="0 0 0"/>',
            '      <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>',
            '    </inertial>',
            '  </link>',
            "",
            f'  <joint name="world_to_box{i}" type="fixed">',
            '    <parent link="world"/>',
            f'    <child link="multi_boxes_box{i}_link"/>',
            '    <origin rpy="0 0 0" xyz="0 0 0"/>',
            '  </joint>',
            "",
        ]
    parts.append("</robot>")
    with open(path, "w") as fh:
        fh.write("\n".join(parts) + "\n")


def write_combined_urdf(path):
    content = '''<?xml version="1.0"?>
<robot name="multi_boxes_combined">
  <link name="multi_boxes_combined_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="multi_boxes.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="multi_boxes.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="100.0"/>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>
    </inertial>
  </link>
</robot>
'''
    with open(path, "w") as fh:
        fh.write(content)


def main():
    ap = argparse.ArgumentParser(description="Generate an N-box staircase artifact.")
    ap.add_argument("--num_boxes", type=int, default=6,
                    help="Total boxes: (num_boxes-1) risers + 1 elongated top landing.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output artifact directory.")
    args = ap.parse_args()

    n = args.num_boxes
    if n < 1:
        raise SystemExit("--num_boxes must be >= 1.")
    # n == 1 is a single elongated box (one riser whose top doubles as the landing),
    # used for the 1-step staircase dataset. n >= 2 is (n-1) risers + 1 elongated
    # top landing. Boxes 1..k are byte-identical across all n, so a k-step staircase
    # is a clean truncation of the full one (matches the matched motion crop).

    box_dir = os.path.join(args.out_dir, "box_models")
    os.makedirs(box_dir, exist_ok=True)

    all_verts = []
    stairs_meta = []
    for i in range(1, n + 1):
        verts, top_z = box_vertices(i, n)
        all_verts.append(verts)
        name = f"step_{i:02d}" + ("_landing" if i == n else "")
        write_box_obj(os.path.join(box_dir, f"box{i}.obj"), name, verts)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        stairs_meta.append({
            "name": name,
            "bounds_min_m": vmin.tolist(),
            "bounds_max_m": vmax.tolist(),
            "size_m": (vmax - vmin).tolist(),
            "top_z_m": float(top_z),
        })

    write_box_obj  # noqa (silence linters about unused in some envs)
    write_multi_boxes_urdf(os.path.join(args.out_dir, "multi_boxes.urdf"), n)
    write_combined_obj(os.path.join(args.out_dir, "multi_boxes.obj"), "multi_boxes", all_verts)
    write_combined_urdf(os.path.join(args.out_dir, "multi_boxes_combined.urdf"))

    all_pts = np.vstack(all_verts)
    metadata = {
        "generated_by": "scripts/make_n_step_staircase.py",
        "extends": "/move/u/karenvo/Projects/rmr_tracking/artifacts/walk_up_karen_stairs",
        "num_boxes": n,
        "num_risers": n - 1,
        "rise_m": RISE,
        "step1_offset_m": STEP1_OFFSET,
        "run_vector_m": RUN.tolist(),
        "landing_run_scale": LANDING_RUN_SCALE,
        "stairs": stairs_meta,
        "combined_bounds_min_m": all_pts.min(axis=0).tolist(),
        "combined_bounds_max_m": all_pts.max(axis=0).tolist(),
    }
    with open(os.path.join(args.out_dir, "staircase_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    _desc = "single elongated step" if n == 1 else f"{n - 1} risers + landing"
    print(f"[OK] Wrote {n}-box staircase ({_desc}) to {args.out_dir}")
    print(f"     top heights (m): {[round(s['top_z_m'], 5) for s in stairs_meta]}")
    print(f"     combined bounds: min={metadata['combined_bounds_min_m']}")
    print(f"                      max={metadata['combined_bounds_max_m']}")


if __name__ == "__main__":
    main()
