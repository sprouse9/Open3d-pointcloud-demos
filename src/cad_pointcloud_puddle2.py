import numpy as np
import open3d as o3d

# -------------------------------
# helpers
# -------------------------------
def gaussian_bump(points, center=(0.5, 0.5), A=0.003, sigma=0.15, z_gate=0.98):
    """Add a smooth bulge/dent (A>0 bulge, A<0 dent) near the top face (z≈1)."""
    cx, cy = center
    r2 = (points[:,0] - cx)**2 + (points[:,1] - cy)**2
    bump = A * np.exp(-r2 / (2*sigma**2))
    mask = points[:,2] > z_gate
    points[mask, 2] += bump[mask]
    return points

def dev_to_rgb(d, tol=0.002):
    """Deviation -> RGB (green near 0, red for +, blue for -)."""
    n = np.clip(d / tol, -1.0, 1.0)
    if n >= 0:
        r, g, b = n, 1.0 - 0.6*n, 0.0
    else:
        n = -n
        r, g, b = 0.0, 1.0 - 0.6*n, n
    return [float(r), float(g), float(b)]

def nearest_face_and_normal(v):
    """For a vertex v on [0,1]^3, pick the nearest axis-aligned face and its outward normal."""
    x, y, z = v
    dists = np.array([-x, x-1.0, -y, y-1.0, -z, z-1.0])  # signed distances to x=0, x=1, y=0, y=1, z=0, z=1
    i = np.argmin(np.abs(dists))
    normals = np.array([
        [-1, 0, 0],  # x=0   -> "X0"
        [ 1, 0, 0],  # x=1   -> "X1"
        [ 0,-1, 0],  # y=0   -> "Y0"
        [ 0, 1, 0],  # y=1   -> "Y1"
        [ 0, 0,-1],  # z=0   -> "Z0"
        [ 0, 0, 1],  # z=1   -> "Z1"
    ], dtype=float)
    names = np.array(["X0","X1","Y0","Y1","Z0","Z1"])
    return dists[i], normals[i], names[i]


# -------------------------------
# 1) CAD cube (nominal) + densify WITHOUT smoothing
# -------------------------------
cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)  # exact [0,1]^3 box
cube.compute_vertex_normals()

# Densify while preserving shape:
cube_dense = o3d.geometry.TriangleMesh(cube)

# Use midpoint subdivision (preserves geometry). If unavailable, set iterations lower or ask me for an alternative.
cube_dense = cube_dense.subdivide_midpoint(number_of_iterations=3)
cube_dense.compute_vertex_normals()

# Lift cube so the origin axes are below it
#cube_dense.translate([1.0, 1.0, 1.0])   # shift 1 unit upward along Z

# -------------------------------
# 2) Measured point cloud: sample + noise + Gaussian bump
# -------------------------------
pcd_nom = cube.sample_points_uniformly(number_of_points=12000)

pcd_meas = o3d.geometry.PointCloud(pcd_nom)
P = np.asarray(pcd_meas.points)

np.random.seed(7)
P += np.random.normal(0.0, 0.0005, size=P.shape)  # ~0.5 mm jitter
P = gaussian_bump(P, A=0.003, sigma=0.15)         # +3 mm bulge on top face
pcd_meas.points = o3d.utility.Vector3dVector(P)

# Build KD-tree for neighbor lookups
kdt = o3d.geometry.KDTreeFlann(pcd_meas)

# -------------------------------
# 3) Color CAD by projecting nearby scan points along the TRUE face normal
# -------------------------------
V = np.asarray(cube_dense.vertices)
colors = np.zeros((len(V), 3))
tol = 0.002      # ±2 mm tolerance
K = 16           # neighbors per vertex for a robust median

devs = np.empty(len(V), dtype=float)          # <-- NEW: signed normal deviations per vertex
face_names = np.empty(len(V), dtype=object)   # <-- NEW: to aggregate per-face stats

for i, v in enumerate(V):
    _, n, fname = nearest_face_and_normal(v)  # unit normal and face name
    _, idx, _ = kdt.search_knn_vector_3d(v, K)
    neigh = P[idx]

    dv = neigh - v
    proj = dv @ n                      # signed projection along the face normal
    d = np.median(proj)                # robust local deviation
    devs[i] = d                        # <-- collect for stats
    face_names[i] = fname

    colors[i] = dev_to_rgb(d, tol=tol)

cube_dense.vertex_colors = o3d.utility.Vector3dVector(colors)

# -------------------------------
# 3.5) Print error summary to terminal
# -------------------------------
absdev = np.abs(devs)
N = devs.size
within = np.sum(absdev <= tol)
over_pos = np.sum(devs >  tol)
over_neg = np.sum(devs < -tol)

mean = float(np.mean(devs))
median = float(np.median(devs))
std = float(np.std(devs, ddof=1))
rms = float(np.sqrt(np.mean(devs**2)))
max_abs = float(np.max(absdev))
p95 = float(np.percentile(absdev, 95))

def hist_ascii(x, bins=21, width=40):
    counts, edges = np.histogram(x, bins=bins)
    barmax = counts.max() if counts.max() > 0 else 1
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(width * c / barmax)
        lines.append(f"{lo:+.4f} – {hi:+.4f} | {bar} ({c})")
    return "\n".join(lines)

print("\n=== Deviation Summary (signed along nearest-face normal) ===")
print(f"N vertices      : {N}")
print(f"Tolerance (±)   : {tol:.4f} units")
print(f"Mean / Median   : {mean:+.5f} / {median:+.5f}")
print(f"Std / RMS       : {std:.5f} / {rms:.5f}")
print(f"Max |dev| / p95 : {max_abs:.5f} / {p95:.5f}")
print(f"Within tol      : {within} / {N}  ({100*within/N:.1f}%)")
print(f"Over + / Over - : {over_pos} / {over_neg}")

# Per-face breakdown
faces = ["X0","X1","Y0","Y1","Z0","Z1"]
print("\n--- Per-face statistics ---")
for f in faces:
    mask = (face_names == f)
    if not np.any(mask):
        continue
    d = devs[mask]
    a = np.abs(d)
    w = np.sum(a <= tol)
    print(f"{f}: N={d.size:5d}  mean={np.mean(d):+.5f}  med={np.median(d):+.5f}  "
          f"rms={np.sqrt(np.mean(d*d)):.5f}  p95={np.percentile(a,95):.5f}  "
          f"within={w}/{d.size} ({100*w/d.size:.1f}%)")

print("\n--- Histogram of signed deviations ---")
print(hist_ascii(devs, bins=25, width=50))

# -------------------------------
# 4) Visualize ONLY the colored CAD (solid puddle plot)
# -------------------------------
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
o3d.visualization.draw_geometries(
    [cube_dense, axes],
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)
