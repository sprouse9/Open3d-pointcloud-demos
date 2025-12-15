# pipe_elbow_fit.py
# End-to-end mini "EB-style" training example:
# CAD truth (vertical cylinder + quarter-torus elbow) -> noisy scan -> ICP -> diameter fit -> deviation heatmap

import numpy as np
import open3d as o3d
from copy import deepcopy

# -----------------------------
# Tunable parameters
# -----------------------------
R_PIPE = 50.0           # mm (tube radius, think OD center surface)
H_VERT = 600.0          # mm vertical rise before elbow sweep
RC_ELBOW = 250.0        # mm elbow centerline radius (quarter bend)
N_CYL = 8000            # points sampled on cylinder
N_TOR = 12000           # points sampled on elbow
SCAN_NOISE_SIGMA = 0.8  # mm Gaussian noise applied to scan
OCCLUDE_FRAC = 0.15     # fraction of scan points randomly dropped (occlusion)
ICP_THRESHOLD = 5.0     # mm correspondence threshold
MAX_ITER = 80           # ICP iterations
SEED = 7                # reproducibility

# Optional known misalignment to inject BEFORE ICP (to test recovery)
INJECT_TRANSLATION_MM = np.array([0.0, 0.0, 0.0])  # e.g., [3.0, -2.0, 1.0]
INJECT_YAW_DEG = 0.0                              # rotation about Z (deg), e.g., 0.8

np.random.seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
def safe_clone(geom):
    """Return a geometry copy that works across Open3D versions."""
    return geom.clone() if hasattr(geom, "clone") else deepcopy(geom)

def rotz(deg):
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    T = np.eye(4)
    T[:3,:3] = R
    return T

def cylinder_mesh(radius, height, z0=0.0, resolution=128):
    """Open3D cylinder is centered at z=0; shift it so bottom is at z0."""
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height,
                                                    resolution=resolution, split=1)
    cyl.compute_vertex_normals()
    cyl.translate((0, 0, z0 + height/2.0))
    return cyl

def quarter_torus_mesh(r_tube, r_center, u_steps=200, v_steps=64):
    """
    Build a quarter torus via parametric surface then place it so that
    its lowest Z point sits at Z=H_VERT (i.e., attaches at top of vertical pipe)
    and turns toward +X.
    """
    us = np.linspace(0, np.pi/2, u_steps)
    vs = np.linspace(0, 2*np.pi, v_steps)
    UU, VV = np.meshgrid(us, vs, indexing="ij")

    # Base torus with centerline in XZ plane, sweeping around Y:
    X = (r_center + r_tube * np.cos(VV)) * np.cos(UU)
    Y = r_tube * np.sin(VV)
    Z = (r_center + r_tube * np.cos(VV)) * np.sin(UU)

    # Swap axes so +X becomes +Z and +Z becomes +X (orient bend upward then toward +X)
    Xp, Yp, Zp = Z, Y, X

    # Lift torus so its minimum Z aligns to H_VERT (meeting the vertical rise)
    z_offset = H_VERT - float(np.min(Zp))
    Zp = Zp + z_offset

    vertices = np.vstack([Xp.ravel(), Yp.ravel(), Zp.ravel()]).T
    tris = []
    for i in range(u_steps - 1):
        for j in range(v_steps - 1):
            a = i * v_steps + j
            b = a + 1
            c = (i + 1) * v_steps + j
            d = c + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
        # wrap last column
        a = i * v_steps + (v_steps - 1)
        b = i * v_steps
        c = (i + 1) * v_steps + (v_steps - 1)
        d = (i + 1) * v_steps
        tris.append([a, c, b])
        tris.append([b, c, d])

    tor = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    )
    tor.compute_vertex_normals()
    return tor

def mat_to_rpy_zyx(T):
    """Return roll, pitch, yaw (deg) for ZYX convention from a 4x4 transform."""
    R = T[:3, :3]
    yaw = np.degrees(np.arctan2(R[1,0], R[0,0]))
    pitch = np.degrees(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)))
    roll = np.degrees(np.arctan2(R[2,1], R[2,2]))
    return roll, pitch, yaw

# -----------------------------
# 1) Build CAD truth: cylinder + quarter torus
# -----------------------------
cyl = cylinder_mesh(R_PIPE, H_VERT, z0=0.0, resolution=96)
tor = quarter_torus_mesh(R_PIPE, RC_ELBOW, u_steps=200, v_steps=64)
cad = cyl + tor
cad.compute_vertex_normals()

# -----------------------------
# 2) Sample a synthetic scan, add noise and occlusion, inject known misalignment
# -----------------------------
pc_clean = cad.sample_points_uniformly(number_of_points=N_CYL + N_TOR)
pts = np.asarray(pc_clean.points)

# Add Gaussian noise (mm)
pts_noisy = pts + np.random.normal(0.0, SCAN_NOISE_SIGMA, size=pts.shape)

# Random occlusion
N = pts_noisy.shape[0]
keep = np.ones(N, dtype=bool)
drop_idx = np.random.choice(N, size=int(OCCLUDE_FRAC * N), replace=False)
keep[drop_idx] = False
pts_scan = pts_noisy[keep]

# Inject known misalignment (translation + yaw) BEFORE ICP to test recovery
T_inject = np.eye(4)
T_inject[:3, 3] = INJECT_TRANSLATION_MM
T_inject = rotz(INJECT_YAW_DEG) @ T_inject
homog = np.hstack([pts_scan, np.ones((pts_scan.shape[0], 1))])
pts_scan = (homog @ T_inject.T)[:, :3]

pc_scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_scan))
pc_scan.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

# -----------------------------
# 3) ICP placement (scan -> CAD)
# -----------------------------
pc_target = cad.sample_points_poisson_disk(50000)
pc_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

reg = o3d.pipelines.registration.registration_icp(
    pc_scan, pc_target, ICP_THRESHOLD, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
)
T_scan_to_cad = reg.transformation

# Pose reported relative to the injected error (what ICP "undid")
# If you injected a transform, the residual T * T_inject should ~ Identity
T_residual = T_scan_to_cad @ T_inject
t_mm = T_scan_to_cad[:3, 3]
r_deg = mat_to_rpy_zyx(T_scan_to_cad)

print("\n=== ICP Placement ===")
print(f"Injected translation (mm): {INJECT_TRANSLATION_MM}")
print(f"Injected yaw (deg): {INJECT_YAW_DEG}")
print("Recovered translation (mm):", np.round(t_mm, 3))
print("Recovered rotation (deg) [roll, pitch, yaw]:", (round(r_deg[0],3), round(r_deg[1],3), round(r_deg[2],3)))
print("ICP fitness:", round(reg.fitness, 3), " | inlier RMSE:", round(reg.inlier_rmse, 3))

# Registered scan (do not assign transform's return value; it mutates in place)
pc_scan_reg = safe_clone(pc_scan)
pc_scan_reg.transform(T_scan_to_cad)

# -----------------------------
# 4) Diameter & center fit on vertical section
# -----------------------------
scan_pts_reg = np.asarray(pc_scan_reg.points)

# Focus below elbow start to avoid curvature (a bit margin)
z_cut = H_VERT - R_PIPE * 0.5
mask_vert = scan_pts_reg[:, 2] <= z_cut
vert_xy = scan_pts_reg[mask_vert][:, :2]

def fit_circle_algebraic(xy):
    # Fit (x - a)^2 + (y - b)^2 = r^2 algebraically: x^2 + y^2 + Dx + Ey + F = 0
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    a, b_ = -D/2.0, -E/2.0
    r = np.sqrt(max((a*a + b_*b_) - F, 0.0))
    return a, b_, r

if vert_xy.shape[0] > 30:
    a, b_, r_est = fit_circle_algebraic(vert_xy)
    center_offset = float(np.hypot(a, b_))
    r_err = r_est - R_PIPE
    print("\n=== Diameter & Placement (vertical run) ===")
    print(f"Estimated center (mm): a={a:.3f}, b={b_:.3f}  |  offset={center_offset:.3f} mm")
    print(f"Estimated radius (mm): {r_est:.3f}  |  True: {R_PIPE:.3f}  |  Error: {r_err:.3f} mm")
else:
    print("\n[Info] Not enough vertical points to fit circle; reduce occlusion or noise.")

# -----------------------------
# 5) Deviation heatmap (scan->CAD)
# -----------------------------
pc_truth_dense = cad.sample_points_poisson_disk(80000)
dists = np.asarray(pc_scan_reg.compute_point_cloud_distance(pc_truth_dense))

# Choose visualization scale:
# Use 99th percentile to avoid outliers blowing colors, but report true max explicitly.
dmax_vis = max(1.0, float(np.percentile(dists, 99)))
dmax_true = float(np.max(dists))
vals = np.clip(dists / dmax_vis, 0.0, 1.0)

def jetish(v):
    # piecewise RGB (blue->cyan->green->yellow->red), v in [0,1]
    v = np.asarray(v)
    r = np.clip(1.5*v - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2*v - 1.0), 0, 1)
    b = np.clip(1.0 - 1.5*v + 0.5, 0, 1)
    return np.vstack([r, g, b]).T

colors = jetish(vals)

print("\n=== Deviation Stats (scan vs CAD) ===")
print("Mean abs dev (mm):", round(float(np.mean(dists)), 3))
print("95th pct dev (mm):", round(float(np.percentile(dists, 95)), 3))
print(f"Max dev (true)  (mm): {dmax_true:.3f}")
print(f"Color scale (99th pct) (mm): {dmax_vis:.2f}")

pc_colored = o3d.geometry.PointCloud()
pc_colored.points = o3d.utility.Vector3dVector(scan_pts_reg)
pc_colored.colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# 6) Visualize
# -----------------------------
cad_vis = safe_clone(cad)
cad_vis.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries(
    [cad_vis, pc_colored],
    window_name="CAD (gray) + Registered Scan (colored by deviation)"
)
