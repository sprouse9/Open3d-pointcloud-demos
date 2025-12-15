import numpy as np
import open3d as o3d

# ---------------------------
# 1) CAD cube (nominal mesh)
# ---------------------------
cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)  # [0,1]^3
cube.compute_vertex_normals()
cube.paint_uniform_color([0.75, 0.75, 0.75])  # light gray

# ----------------------------------------
# 2) Nominal point cloud from cube faces
# ----------------------------------------
pcd_nominal = cube.sample_points_uniformly(number_of_points=6000)

# ---------------------------------------------------
# 3) Make a "measured" scan with a Gaussian bump
#     + small sensor noise
#     Bump on top face (near z ~ 1), centered at (0.5, 0.5)
# ---------------------------------------------------
pcd_meas = o3d.geometry.PointCloud(pcd_nominal)  # copy
P = np.asarray(pcd_meas.points)

# (a) Sensor jitter (e.g., 0.5 mm std dev on a 1 m cube)
np.random.seed(7)
P += np.random.normal(0.0, 0.0005, size=P.shape)

# (b) Smooth "manufacturing" bump on top face
#     amplitude A (meters), radius via sigma (meters)
A = 0.003  # 3 mm bump outward (+z)
sigma = 0.15
cx, cy = 0.5, 0.5
r2 = (P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2
bump = A * np.exp(-r2 / (2 * sigma ** 2))

# Only push points that were originally near the top face
# (close to z=1 before noise). Threshold controls "patch" thickness.
top_mask = P[:, 2] > 0.98
P[top_mask, 2] += bump[top_mask]

pcd_meas.points = o3d.utility.Vector3dVector(P)

# --------------------------------------------------
# 4) Compute signed distance to nearest cube face
#     For [0,1]^3, planes are:
#     x=0 (normal -x):  d0 = -x
#     x=1 (normal +x):  d1 =  x - 1
#     y=0 (normal -y):  d2 = -y
#     y=1 (normal +y):  d3 =  y - 1
#     z=0 (normal -z):  d4 = -z
#     z=1 (normal +z):  d5 =  z - 1
#   The deviation is the signed distance to the *nearest* face.
# --------------------------------------------------
def signed_face_distance(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d = np.stack([
        -x,       # to x=0
         x - 1.0, # to x=1
        -y,       # to y=0
         y - 1.0, # to y=1
        -z,       # to z=0
         z - 1.0  # to z=1
    ], axis=1)  # shape (N,6)
    # choose the face with the smallest absolute distance
    idx = np.argmin(np.abs(d), axis=1)
    return d[np.arange(len(points)), idx]

dev = signed_face_distance(P)  # signed deviation (meters)

# --------------------------------------------------
# 5) Color by deviation vs tolerance
#     Green inside tol, yellow near edge, red/blue outside
# --------------------------------------------------
tol = 0.002  # ±2 mm tolerance
# normalize and clamp for coloring
n = np.clip(dev / tol, -2.0, 2.0)  # -2..2

# simple red/green/blue mapping:
#  n <= -1: deep blue,  n ~ 0: green,  n >= +1: deep red
colors = np.zeros((len(n), 3))
# negative deviations (inward) → blue→green blend
neg = n < 0
colors[neg, 2] = np.clip(-n[neg], 0, 1)           # blue channel
colors[neg, 1] = 1.0 - colors[neg, 2] * 0.6       # keep some green
# positive deviations (outward) → green→red blend
pos = n > 0
colors[pos, 0] = np.clip(n[pos], 0, 1)            # red channel
colors[pos, 1] = 1.0 - colors[pos, 0] * 0.6       # keep some green
# near zero (~green already)

pcd_meas.colors = o3d.utility.Vector3dVector(colors)

# --------------------------------------------------
# 6) Visualize: CAD cube + colored measured points
# --------------------------------------------------
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)

# show triangle edges too (quick wireframe)
o3d.visualization.draw_geometries(
    [cube, pcd_meas, axes],
    mesh_show_wireframe=True,
    mesh_show_back_face=True
)
