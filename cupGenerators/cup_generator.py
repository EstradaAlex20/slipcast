"""
cup_generator.py
Generate a watertight hollow cup as an STL file for 3D printing.

Adjust the parameters at the top of the file, then run:
    python cup_generator.py
"""

import numpy as np
import trimesh
from noise import pnoise3

# ── Parameters ──────────────────────────────────────────────────────────────
BOT_OUTER_RADIUS = 30.0   # mm — outer radius at the base (narrower)
TOP_OUTER_RADIUS = 45.0   # mm — outer radius at the rim  (wider)
HEIGHT           = 100.0  # mm — total cup height
WALL_THICKNESS   =  3.0   # mm — side wall thickness (kept smooth on the inside)
FLOOR_THICKNESS  =  3.0   # mm — base plate thickness
SECTIONS         = 256     # vertices per ring (more = smoother circle)
WALL_SEGMENTS    = 256     # vertical ring count on the outer wall (more = smoother noise)

# ── Noise parameters ─────────────────────────────────────────────────────────
# The outer wall is displaced outward by 3D Perlin noise. The angle maps to
# a circle in the XY noise plane so the texture wraps seamlessly at 360°.
NOISE_AMPLITUDE  =  3.0   # mm — how far the surface can bulge (approximate max)
NOISE_SCALE      =  2.0   # frequency: ~how many bumps fit around/along the cup
NOISE_OCTAVES    =  4     # layers of detail (higher = rougher, more complex)

OUTPUT_FILE = "cup.stl"
# ────────────────────────────────────────────────────────────────────────────


def make_cup(
    bot_outer_radius, top_outer_radius,
    height, wall_thickness, floor_thickness,
    sections, wall_segments,
    noise_amplitude, noise_scale, noise_octaves,
):
    """
    Build a hollow tapered cup with Perlin noise on the outer wall.

    Five surface regions (same as before):
        1. Base cap      — solid circle at z=0,              normal −z
        2. Outer wall    — subdivided into wall_segments rings, normal +r (noisy)
        3. Rim           — annulus at z=height,              normal +z
        4. Inner wall    — smooth cylinder inside,           normal −r
        5. Inner floor   — solid circle at z=floor_thickness, normal +z
    """
    n  = sections
    S  = wall_segments          # number of ring-to-ring steps on the outer wall
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

    # ── Ring builders ─────────────────────────────────────────────────────
    def ring(radius, z):
        """Plain circular ring at a fixed radius and height."""
        return np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)]
        )

    def noisy_ring(base_radius, z, z_frac):
        """
        Ring whose radius varies per-vertex using 3D Perlin noise.

        To avoid a seam at 0°/360°, the angle is mapped onto a circle in
        the XY plane of noise space: nx=cos(a)*scale, ny=sin(a)*scale.
        As the angle completes a full revolution the noise inputs trace a
        closed loop, so the first and last vertex always match.
        The height maps to the Z axis of noise space.
        """
        radii = np.array([
            base_radius + pnoise3(
                np.cos(a) * noise_scale,   # x — angular position on noise circle
                np.sin(a) * noise_scale,   # y — angular position on noise circle
                z_frac    * noise_scale,   # z — height position
                octaves=noise_octaves,
            ) * noise_amplitude
            for a in angles
        ])
        return np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles), np.full(n, z)]
        )

    # ── Vertices ──────────────────────────────────────────────────────────
    # Outer wall: S+1 rings from z=0 to z=H, each with Perlin displacement.
    outer_rings = []
    for k in range(S + 1):
        z_frac   = k / S
        z        = z_frac * H
        base_r   = R_bot + (R_top - R_bot) * z_frac   # linear taper baseline
        outer_rings.append(noisy_ring(base_r, z, z_frac))

    # Inner wall and floor: smooth (no noise — texture is only on the outside).
    inner_top  = ring(r_top, H)
    inner_bot  = ring(r_bot, t)

    # Vertex layout in the final array:
    #   outer_rings[0..S] : (S+1)*n vertices
    #   inner_top         : n vertices   at offset (S+1)*n
    #   inner_bot         : n vertices   at offset (S+2)*n
    #   bot_center        : 1 vertex     at offset (S+3)*n
    #   flr_center        : 1 vertex     at offset (S+3)*n + 1
    vertices = np.vstack([
        *outer_rings,
        inner_top,
        inner_bot,
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, t]],
    ])

    # Index helpers
    def outer_idx(k):
        """Index array for outer ring k (0 = base, S = top)."""
        return np.arange(k * n, (k + 1) * n)

    idx_inner_top = np.arange((S + 1) * n, (S + 2) * n)
    idx_inner_bot = np.arange((S + 2) * n, (S + 3) * n)
    idx_bot_ctr   = (S + 3) * n
    idx_flr_ctr   = (S + 3) * n + 1

    # ── Face helpers (unchanged from before) ──────────────────────────────
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
    cup = make_cup(
        bot_outer_radius=BOT_OUTER_RADIUS,
        top_outer_radius=TOP_OUTER_RADIUS,
        height=HEIGHT,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        sections=SECTIONS,
        wall_segments=WALL_SEGMENTS,
        noise_amplitude=NOISE_AMPLITUDE,
        noise_scale=NOISE_SCALE,
        noise_octaves=NOISE_OCTAVES,
    )

    cup.export(OUTPUT_FILE)

    vol_ml = cup.volume / 1000.0  # mm³ → cm³ ≈ mL
    print(f"Saved       : {OUTPUT_FILE}")
    print(f"Faces       : {len(cup.faces)}")
    print(f"Volume      : {vol_ml:.1f} mL  (material volume, not capacity)")
    print(f"Watertight  : {cup.is_watertight}")
    if not cup.is_watertight:
        print("WARNING: mesh has holes — slicer may reject it. Check parameters.")
