"""
ribbed_cup_generator.py
Generate a tapered cup with evenly-spaced raised vertical ridges on the
outside. A Perlin-noise flow field makes the ridges drift left and right
as they travel up the cup.

Adjust the parameters at the top of the file, then run:
    python ribbed_cup_generator.py
"""

import numpy as np
import trimesh
from noise import pnoise3

# ── Cup shape ────────────────────────────────────────────────────────────────
BOT_OUTER_RADIUS = 30.0   # mm — outer radius at the base (narrower)
TOP_OUTER_RADIUS = 45.0   # mm — outer radius at the rim  (wider)
HEIGHT           = 100.0  # mm — total cup height
WALL_THICKNESS   =  3.0   # mm — side wall thickness (inner surface stays smooth)
FLOOR_THICKNESS  =  3.0   # mm — base plate thickness
SECTIONS         = 4000    # vertices per ring — needs to be high enough to
                          # resolve the ridges (aim for ~20+ vertices per ridge)
WALL_SEGMENTS    = 1000     # vertical subdivisions — more needed now so the
                          # flowing curves look smooth along the height

# ── Ridge parameters ─────────────────────────────────────────────────────────
NUM_RIDGES   = 128    # how many raised lines around the circumference
RIDGE_HEIGHT =  1.5  # mm — how far each ridge protrudes from the base surface
RIDGE_SHAPE  =  1.0  # sharpness exponent:
                     #   1 = broad, sine-wave bumps
                     #   3 = medium — rounded ridges with flat valleys
                     #   6 = narrow, sharp ridges with wide flat valleys

# ── Flow field parameters ────────────────────────────────────────────────────
# A Perlin noise field is integrated upward from z=0 to z=H.  At each height
# the field value at each angular position is used as a lateral drift rate,
# so ridges are deflected left or right by a coherent, organic amount.
FLOW_SCALE    = 2.0   # spatial frequency of the flow field — lower = broader
                      # sweeping curves, higher = more tightly wound drift
FLOW_STRENGTH = 2.0   # total angular drift budget (radians). ~0.5 = subtle
                      # lean, ~1.5 = pronounced S-curves, >3 = swirling

OUTPUT_FILE = "ribbed_cup.stl"
# ────────────────────────────────────────────────────────────────────────────


def make_ribbed_cup(
    bot_outer_radius, top_outer_radius,
    height, wall_thickness, floor_thickness,
    sections, wall_segments,
    num_ridges, ridge_height, ridge_shape,
    flow_scale, flow_strength,
):
    """
    Build a hollow tapered cup with Perlin-flow-distorted vertical ridges.

    Ridge displacement at ring k, vertex i:
        sample_angle = angles[i] + angular_offset[k, i]
        displacement = ridge_height * max(0, cos(sample_angle * num_ridges)) ^ ridge_shape

    angular_offset is built by integrating the flow field from z=0 upward:
        angular_offset[k] = angular_offset[k-1]
                          + pnoise3(cos(a)*scale, sin(a)*scale, z*scale)
                          * flow_strength * dz

    The cos/sin mapping of the angle into noise space keeps the field
    seamlessly periodic at 0°/360° — the same trick used in cup_generator.py.

    Five surface regions (same structure as cup_generator.py):
        1. Base cap      — z=0,              normal −z
        2. Outer wall    — flow-distorted,   normal +r
        3. Rim           — z=height,         normal +z
        4. Inner wall    — smooth,           normal −r
        5. Inner floor   — z=floor_thickness, normal +z
    """
    n  = sections
    S  = wall_segments
    R_bot = bot_outer_radius
    R_top = top_outer_radius
    r_bot = R_bot - wall_thickness
    r_top = R_top - wall_thickness
    H  = height
    t  = floor_thickness
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if r_bot <= 0 or r_top <= 0:
        raise ValueError("wall_thickness must be less than both radii")
    if t >= H:
        raise ValueError("floor_thickness must be less than height")

    # ── Flow field integration ─────────────────────────────────────────────
    # Rather than shifting the ridge sampling phase per-vertex (which distorts
    # ridge shapes), we track the angular position of each ridge center as it
    # flows up the cup. Each ridge is a particle that moves through the noise
    # field independently, sampled at its own current angle.
    #
    # ridge_pos[k, r] = angular position of ridge r at ring k.
    dz = 1.0 / S
    ridge_pos = np.zeros((S + 1, num_ridges))
    ridge_pos[0] = np.linspace(0, 2 * np.pi, num_ridges, endpoint=False)

    for k in range(1, S + 1):
        z_frac = k / S
        for r in range(num_ridges):
            a = ridge_pos[k - 1, r]
            drift = pnoise3(
                np.cos(a) * flow_scale,
                np.sin(a) * flow_scale,
                z_frac    * flow_scale,
            )
            ridge_pos[k, r] = ridge_pos[k - 1, r] + drift * flow_strength * dz

    # ── Ring builders ─────────────────────────────────────────────────────
    def ring(radius, z):
        """Plain circular ring — used for the smooth inner surfaces."""
        return np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)]
        )

    def ridged_ring(base_radius, z, k):
        """
        For each vertex, find its angular distance to the nearest ridge center
        at this height, then apply the ridge bump profile based on that distance.

        Because we measure distance to the actual ridge center positions (not a
        phase-shifted version of the whole pattern), each ridge keeps its correct
        shape as it drifts — only its angular position changes, never its width
        or height. No spikes from phase compression.
        """
        # diffs[i, r] = angular distance from vertex i to ridge r's center.
        # Broadcasting: angles is (n,), ridge_pos[k] is (num_ridges,).
        diffs = (angles[:, np.newaxis] - ridge_pos[k][np.newaxis, :]) % (2 * np.pi)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)   # shortest arc, in [0, π]
        min_dists = diffs.min(axis=1)                   # closest ridge, shape (n,)

        # cos maps distance=0 (on center) → 1, distance=π/num_ridges (midway) → −1.
        # Clamping at 0 gives a clean bump that never indents the surface.
        displacement = (
            np.maximum(0.0, np.cos(min_dists * num_ridges)) ** ridge_shape
            * ridge_height
        )
        radii = base_radius + displacement
        return np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles), np.full(n, z)]
        )

    # ── Vertices ──────────────────────────────────────────────────────────
    # Outer wall: S+1 rings from z=0 to z=H with flow-distorted ridges.
    outer_rings = []
    for k in range(S + 1):
        z_frac = k / S
        z      = z_frac * H
        base_r = R_bot + (R_top - R_bot) * z_frac   # linear taper
        outer_rings.append(ridged_ring(base_r, z, k))

    inner_top = ring(r_top, H)
    inner_bot = ring(r_bot, t)

    vertices = np.vstack([
        *outer_rings,          # (S+1)*n vertices
        inner_top,             # n vertices  at (S+1)*n
        inner_bot,             # n vertices  at (S+2)*n
        [[0.0, 0.0, 0.0]],    # 1 vertex    at (S+3)*n
        [[0.0, 0.0, t]],      # 1 vertex    at (S+3)*n + 1
    ])

    def outer_idx(k):
        return np.arange(k * n, (k + 1) * n)

    idx_inner_top = np.arange((S + 1) * n, (S + 2) * n)
    idx_inner_bot = np.arange((S + 2) * n, (S + 3) * n)
    idx_bot_ctr   = (S + 3) * n
    idx_flr_ctr   = (S + 3) * n + 1

    # ── Face helpers ──────────────────────────────────────────────────────
    def wall_quads(ring_a, ring_b):
        tris = []
        for i in range(n):
            j = (i + 1) % n
            a0, a1 = ring_a[i], ring_a[j]
            b0, b1 = ring_b[i], ring_b[j]
            tris.append([a0, b1, b0])
            tris.append([a0, a1, b1])
        return tris

    def cap_fan(center_idx, ring_idx, flip):
        tris = []
        for i in range(n):
            j = (i + 1) % n
            if flip:
                tris.append([center_idx, ring_idx[j], ring_idx[i]])
            else:
                tris.append([center_idx, ring_idx[i], ring_idx[j]])
        return tris

    # ── Assemble faces ────────────────────────────────────────────────────
    faces = cap_fan(idx_bot_ctr, outer_idx(0), flip=True)   # 1. base cap

    for k in range(S):                                       # 2. outer wall
        faces += wall_quads(outer_idx(k), outer_idx(k + 1))

    faces += wall_quads(outer_idx(S), idx_inner_top)        # 3. rim
    faces += wall_quads(idx_inner_top, idx_inner_bot)       # 4. inner wall
    faces += cap_fan(idx_flr_ctr, idx_inner_bot, flip=False) # 5. inner floor

    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )


if __name__ == "__main__":
    cup = make_ribbed_cup(
        bot_outer_radius=BOT_OUTER_RADIUS,
        top_outer_radius=TOP_OUTER_RADIUS,
        height=HEIGHT,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        sections=SECTIONS,
        wall_segments=WALL_SEGMENTS,
        num_ridges=NUM_RIDGES,
        ridge_height=RIDGE_HEIGHT,
        ridge_shape=RIDGE_SHAPE,
        flow_scale=FLOW_SCALE,
        flow_strength=FLOW_STRENGTH,
    )

    cup.export(OUTPUT_FILE)

    vol_ml = cup.volume / 1000.0
    print(f"Saved       : {OUTPUT_FILE}")
    print(f"Faces       : {len(cup.faces)}")
    print(f"Volume      : {vol_ml:.1f} mL  (material volume, not capacity)")
    print(f"Watertight  : {cup.is_watertight}")
    if not cup.is_watertight:
        print("WARNING: mesh has holes — slicer may reject it. Check parameters.")
