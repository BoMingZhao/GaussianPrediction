# import open3d as o3d
import numpy as np
import torch
# import open3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
import trimesh
from kmeans_pytorch import kmeans
from torch_scatter import scatter
import open3d as o3d

class Visualizer:
    def __init__(self, extri, intri):
        if torch.is_tensor(extri):
            extri = extri.cpu().numpy()
        if torch.is_tensor(intri):
            intri = intri.cpu().numpy()
        self.w2c = extri
        self.k = intri

    def draw_trajectory(self, trajectory):
        if torch.is_tensor(trajectory):
            trajectory = trajectory.cpu().numpy()
        if hasattr(self, 'pcd'):
            pass
        else:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(trajectory)
            self.pcd.transform(self.w2c)

            colors = np.random.uniform(0, 1, size=(trajectory.shape[0], 3))
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

def cluster(features, method="KMeans", k=4):
    if method == "KMeans":
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        labels = kmeans.labels_
    else:
        meanShift = MeanShift()
        meanShift.fit(features)
        labels = meanShift.labels_
    return labels

def draw_motion_feature(points, features, output_path="vis_feature.txt", down_dim=1):
    pca = PCA(n_components=down_dim)
    features_reduced = pca.fit_transform(features)
    print(features_reduced.shape)
    cmap = plt.cm.get_cmap('jet')
    # features_color = cmap(features_reduced[:, 0] / np.max(features_reduced[:, 0]))
    features_color = cmap(features_reduced[:, 0])
    points_vis = np.concatenate([points, features_color], axis=-1)
    np.savetxt(output_path, points_vis)


def PCA_vis(xyzs, features, output_path="default.ply", dim=3, return_only=False, finish_return=False):
    xyzs = xyzs.cpu().numpy()
    pca = PCA(dim, random_state=42)
    f_samples = features.cpu().numpy() if dim == 3 else np.concatenate([xyzs, features.cpu().numpy()], axis=-1)
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = f_samples.mean(0)
    feature_pca_components = pca.components_
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)

    vis_feature = (f_samples - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    if dim > 3:
        xyzs = vis_feature[..., :2]
        vis_feature = vis_feature[..., 2:]
        z = np.ones_like(xyzs)
        xyzs = np.concatenate([xyzs, z[..., 0:1]],axis=-1)
    vis_feature = vis_feature.clip(0.0, 1.0).astype(np.float32)

    if return_only:
        return xyzs, vis_feature
    pcd = trimesh.PointCloud(xyzs, vis_feature)
    pcd.export(output_path)
    if finish_return:
        return xyzs, vis_feature

def feature_kmeans(xyzs, features, K=10, vis=False, return_idx=False):
    cluster_ids_x, cluster_centers = kmeans(X=features, num_clusters=K, device=features.device)
    xyzs_means = scatter(xyzs, cluster_ids_x.to(xyzs.device), dim=0, reduce="mean")

    if vis:
        PCA_vis(xyzs_means, cluster_centers, "keypoints.ply")
        # PCA_vis(self._xyz, cluster_centers, "keypoints.ply")
    if return_idx:
        return xyzs_means, cluster_centers.to(xyzs.device), cluster_ids_x
    return xyzs_means, cluster_centers.to(xyzs.device) 

def draw_weights(weights_xyz, pcd, super_gaussians, idx):
    weights_xyz = weights_xyz[:, idx].cpu().numpy()
    pcd_ = pcd.cpu().numpy()
    super_gaussians = super_gaussians.cpu().numpy()[..., :3][idx].reshape([1, 3]).repeat(3, axis=0)
    color = plt.cm.jet(weights_xyz[None]).reshape([pcd_.shape[0], -1])[..., :3]
    pcd_ = np.concatenate([pcd_, color], axis=-1)
    np.savetxt(f"visualize/weights_xyz_{idx}.txt", pcd_)
    np.savetxt(f"visualize/kpts{idx}.txt", super_gaussians)

def draw_trajectory(pcd, start_xyz, final_xyz):
    pcd = pcd.cpu().numpy()
    start_xyz = start_xyz.cpu().numpy()
    final_xyz = final_xyz.cpu().numpy()
    point_cloud = o3d.geometry.PointCloud()
    num_points = pcd.shape[1]
    num_frames = pcd.shape[0]
    trajectory_colors = np.random.uniform(0, 1, (num_points, 3))
    black_color = np.array([0., 0., 0.]).reshape([1, 3]).repeat(start_xyz.shape[0], axis=0)
    shape_sizes = np.array([0.5]).reshape([1]).repeat(2 * start_xyz.shape[0], axis=0)
    trajectory_sizes = np.array([1.5]).reshape([1]).repeat(num_points * num_frames, axis=0)
    points = []
    colors = []

    for point_index in range(num_points):
        for frame in range(num_frames):
            # 在每个帧上生成点的位置
            x = pcd[frame, point_index, 0]
            y = pcd[frame, point_index, 1]
            z = pcd[frame, point_index, 2]
            
            points.append([x, y, z])
            colors.append(trajectory_colors[point_index])

    # 创建可视化窗口并添加点云
    points = np.array(points)
    colors = np.array(colors)
    points = np.concatenate([points, start_xyz, final_xyz], axis=0)
    colors = np.concatenate([colors, black_color, black_color], axis=0)
    sizes = np.concatenate([trajectory_sizes, shape_sizes], axis=0)
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])
            