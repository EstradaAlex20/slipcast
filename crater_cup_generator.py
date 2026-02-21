"""
crater_cup_generator.py
Generate a tapered cup covered in Voronoi-cell depressions (craters / divots).

Each crater is centred on a randomly placed seed point on the cylinder surface.
Craters never overlap because the depression profile is defined by Voronoi
proximity: each vertex belongs to exactly one cell (the nearest seed), and the
displacement falls to zero precisely at the cell boundary.

Adjust the parameters at the top, then run:
    python crater_cup_generator.py
"""

import numpy as np
import trimesh

# ── Cup shape ────────────────────────────────────────────────────────────────
BOT_OUTER_RADIUS = 30.0   # mm — outer radius at the base (narrower)
TOP_OUTER_RADIUS = 45.0   # mm — outer radius at the rim  (wider)
HEIGHT           = 100.0  # mm — total cup height
WALL_THICKNESS   =  3.0   # mm — side wall thickness (inner stays smooth)
FLOOR_THICKNESS  =  3.0   # mm — base plate thickness
SECTIONS         = 1000    # vertices per ring
WALL_SEGMENTS    = 1000    # vertical ring subdivisions — needs enough to
                          # capture crater curvature along the height

# ── Crater parameters ─────────────────────────────────────────────────────────
NUM_CELLS    =  200    # number of Voronoi cells (= number of craters)
CRATER_DEPTH =  1   # mm — depth of each pool below the cup surface
RANDOM_SEED  =  53799    # change this for a different random arrangement

# Pool cross-section is split into three zones (fractions of the cell radius):
#
#   ←── POOL_FLOOR ──→← wall →←── POOL_BARRIER ──→
#   ___________________                 ____________
#   flat floor         \               /  flat surface (barrier between pools)
#                       ╲_____________/
#                        rounded wall
#
# POOL_FLOOR + POOL_BARRIER must be < 1.0 (the remainder becomes the wall).
POOL_FLOOR   = 0.25   # fraction of cell radius that is flat pool floor
POOL_BARRIER = 0.25   # fraction of cell radius that is flat barrier between pools

CELL_ROUNDNESS   = 3.0   # mm — smooth-min radius for cell corner rounding.
                          #      0 = hard polygonal Voronoi corners.
                          #      Higher values soften where three cells meet.
BAND_WIDTH       = 5.0   # mm — flat (crater-free) band at the top and bottom of the cup
BAND_TRANSITION  = 8.0   # mm — ramp width inside the crater zone where depth fades
                          #      from 0 → full at the bottom, and full → 0 at the top.
                          #      Keeps wall angles printable by avoiding sudden steps.

OUTPUT_FILE = "crater_cup.stl"
# ────────────────────────────────────────────────────────────────────────────


def _jittered_seeds(R_avg, H, num_cells, rng, z_min=0.0, z_max=None):
    """
    Place seed points on the cylinder using a jittered grid.

    The unwrapped cylinder surface (width = 2π·R_avg, height = z_max-z_min) is
    divided into an nrows × ncols grid of tiles, sized as square as possible.
    One seed is placed at a uniformly random position within each tile.

    z_min / z_max let the caller restrict seeding to a sub-range of the cup
    height (used to keep seeds out of the flat top/bottom bands).

    Returns (seed_theta, seed_z), each of shape (nrows*ncols,).
    """
    if z_max is None:
        z_max = H
    z_range = z_max - z_min

    surface_width = 2 * np.pi * R_avg
    aspect = surface_width / z_range                    # width ÷ active height

    ncols = max(1, round(np.sqrt(num_cells * aspect)))  # keep tiles roughly square
    nrows = max(1, round(num_cells / ncols))

    jitter_theta = rng.uniform(0, 1, (nrows, ncols))
    jitter_z     = rng.uniform(0, 1, (nrows, ncols))

    col_idx = np.arange(ncols)
    row_idx = np.arange(nrows)

    theta = (col_idx[None, :] + jitter_theta) * (2 * np.pi / ncols)
    z     = z_min + (row_idx[:, None] + jitter_z) * (z_range / nrows)

    return theta.ravel(), z.ravel()


def make_crater_cup(
    bot_outer_radius, top_outer_radius,
    height, wall_thickness, floor_thickness,
    sections, wall_segments,
    num_cells, crater_depth, random_seed,
    pool_floor, pool_barrier, band_width, band_transition, cell_roundness,
):
    """
    Build a hollow tapered cup with Voronoi-cell depressions on the outer wall.

    Depression profile at vertex v in cell c:
        d1 = distance(v, nearest seed)
        d2 = distance(v, second-nearest seed)
        t  = d1 / d2                  ← 0 at cell centre, 1 at cell boundary
        displacement = -crater_depth * (1 - t²)

    Distances are measured on the unwrapped cylinder surface (arc-length in the
    angular direction, direct in the height direction), using the average radius
    as the arc-length scale.  Angular wrapping is handled so that seeds near
    θ=0 or θ=2π produce complete craters with no seam.

    Five surface regions (same structure as cup_generator.py):
        1. Base cap      — z=0,                normal −z
        2. Outer wall    — Voronoi-cratered,   normal +r
        3. Rim           — z=height,           normal +z
        4. Inner wall    — smooth,             normal −r
        5. Inner floor   — z=floor_thickness,  normal +z
    """
    n  = sections
    S  = wall_segments
    R_bot = bot_outer_radius
    R_top = top_outer_radius
    r_bot = R_bot - wall_thickness
    r_top = R_top - wall_thickness
    H  = height
    t  = floor_thickness
    R_avg = (R_bot + R_top) / 2.0   # reference radius for arc-length distances

    if r_bot <= 0 or r_top <= 0:
        raise ValueError("wall_thickness must be less than both radii")
    if t >= H:
        raise ValueError("floor_thickness must be less than height")

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # ── Seed points ───────────────────────────────────────────────────────
    # Restrict seeds to [band_width, H - band_width] so no crater is centred
    # inside the flat margin bands at the top and bottom.
    rng = np.random.default_rng(random_seed)
    seed_theta, seed_z = _jittered_seeds(
        R_avg, H, num_cells, rng,
        z_min=band_width, z_max=H - band_width,
    )

    # ── Voronoi displacement ───────────────────────────────────────────────
    def voronoi_displacement(z):
        """
        For every vertex in the ring at height z, compute the inward radial
        displacement based on proximity to the nearest and second-nearest seed.

        Returns a 1-D array of shape (n,) with negative values (inward).
        Rings inside the top/bottom band return all zeros (flat surface).
        """
        # Flat band at top and bottom — return immediately with no displacement
        if z < band_width or z > H - band_width:
            return np.zeros(n)

        # Smooth fade in/out near the band edges so crater depth ramps gradually
        # from 0 → full over `band_transition` mm.  This prevents the abrupt
        # radial step that would create an overhang at the band boundary.
        # Smoothstep: f(s) = 3s² − 2s³  — zero slope at both ends.
        fade = 1.0
        if z < band_width + band_transition:
            s = (z - band_width) / band_transition
            fade = 3*s**2 - 2*s**3
        elif z > H - band_width - band_transition:
            s = (H - band_width - z) / band_transition
            fade = 3*s**2 - 2*s**3

        # Angular arc-length distance from each vertex to each seed: (n, C)
        d_theta = np.abs(angles[:, None] - seed_theta[None, :])
        d_theta = np.minimum(d_theta, 2 * np.pi - d_theta)   # shortest arc
        d_arc   = R_avg * d_theta                             # mm

        # Height distance from this ring to each seed: (C,) → broadcast (n, C)
        d_z = np.abs(z - seed_z)

        # Euclidean distance on the unwrapped cylinder surface: (n, C)
        dists = np.sqrt(d_arc ** 2 + d_z[None, :] ** 2)

        # Two smallest distances per vertex (partial sort — faster than full sort)
        # np.partition(arr, 1) guarantees arr[:,0] ≤ arr[:,1] ≤ rest
        part = np.partition(dists, 1, axis=1)
        d1   = part[:, 0]   # distance to nearest seed       (n,)
        d2   = part[:, 1]   # distance to second-nearest seed (n,)

        # Optional corner rounding via smooth-minimum (Inigo Quilez smin).
        # At a Voronoi vertex (triple-point) d1 ≈ d2, so the standard formula
        # gives t_cell ≈ 1 (flat barrier) from three directions simultaneously,
        # forming a sharp ridge.  The smooth-min blends d1 slightly toward d2
        # near boundaries, pulling those corner points back into the wall zone.
        #
        #   h = clip(0.5 + 0.5*(d2-d1)/k, 0, 1)
        #   smooth_d1 = lerp(d2, d1, h) - k·h·(1-h)
        #
        # k=0 → no change; larger k → wider blending zone, rounder corners.
        if cell_roundness > 0:
            h  = np.clip(0.5 + 0.5 * (d2 - d1) / cell_roundness, 0.0, 1.0)
            d1 = d2 * (1.0 - h) + d1 * h - cell_roundness * h * (1.0 - h)

        # Normalised position within cell: 0 at centre, 1 at Voronoi boundary
        t_cell = d1 / np.maximum(d2, 1e-9)

        # Three-zone pool profile
        # ─────────────────────────────────────────────────────────────────
        # Zone boundaries in t:
        #   [0,          floor_end]  → flat pool floor at full depth
        #   [floor_end,  wall_end]   → rounded wall (cubic Hermite ease)
        #   [wall_end,   1       ]   → flat cup surface (barrier between pools)
        #
        # Cubic Hermite on the wall zone:  s = remap of t to [0, 1]
        #   h(s) = 2s³ - 3s² + 1  →  h(0)=1, h(1)=0, h'(0)=h'(1)=0
        # Zero derivatives at both ends mean the wall meets the flat floor
        # and the flat barrier tangentially — no hard crease anywhere.
        floor_end = pool_floor
        wall_end  = 1.0 - pool_barrier

        disp = np.zeros(len(t_cell))

        mask_floor = t_cell <= floor_end
        disp[mask_floor] = -crater_depth

        mask_wall = (t_cell > floor_end) & (t_cell <= wall_end)
        s = (t_cell[mask_wall] - floor_end) / (wall_end - floor_end)
        disp[mask_wall] = -crater_depth * (2*s**3 - 3*s**2 + 1)

        # Zone 3 (t > wall_end): flat surface, disp stays 0

        return disp * fade

    # ── Ring builders ─────────────────────────────────────────────────────
    def ring(radius, z):
        """Plain circular ring — used for the smooth inner surfaces."""
        return np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)]
        )

    def crater_ring(base_radius, z):
        """Outer ring with Voronoi crater displacements applied inward."""
        radii = base_radius + voronoi_displacement(z)
        return np.column_stack(
            [radii * np.cos(angles), radii * np.sin(angles), np.full(n, z)]
        )

    # ── Vertices ──────────────────────────────────────────────────────────
    outer_rings = []
    for k in range(S + 1):
        z_frac = k / S
        z      = z_frac * H
        base_r = R_bot + (R_top - R_bot) * z_frac   # linear taper baseline
        outer_rings.append(crater_ring(base_r, z))

    inner_top = ring(r_top, H)
    inner_bot = ring(r_bot, t)

    vertices = np.vstack([
        *outer_rings,
        inner_top,
        inner_bot,
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, t]],
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
    cup = make_crater_cup(
        bot_outer_radius=BOT_OUTER_RADIUS,
        top_outer_radius=TOP_OUTER_RADIUS,
        height=HEIGHT,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        sections=SECTIONS,
        wall_segments=WALL_SEGMENTS,
        num_cells=NUM_CELLS,
        crater_depth=CRATER_DEPTH,
        random_seed=RANDOM_SEED,
        pool_floor=POOL_FLOOR,
        pool_barrier=POOL_BARRIER,
        band_width=BAND_WIDTH,
        band_transition=BAND_TRANSITION,
        cell_roundness=CELL_ROUNDNESS,
    )

    cup.export(OUTPUT_FILE)

    vol_ml = cup.volume / 1000.0
    print(f"Saved       : {OUTPUT_FILE}")
    print(f"Faces       : {len(cup.faces)}")
    print(f"Volume      : {vol_ml:.1f} mL  (material volume, not capacity)")
    print(f"Watertight  : {cup.is_watertight}")
    if not cup.is_watertight:
        print("WARNING: mesh has holes — slicer may reject it. Check parameters.")
