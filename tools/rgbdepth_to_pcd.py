import os
import numpy as np
import open3d as o3d


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# 返回相机坐标系下的点云
def point_cloud_camera(depth_map, intrinsic_matrix, rgb_image=None, downsample_factor=1):
    """
    将深度图转换为相机坐标系下的点云
    :param depth_map: 深度图 (H, W)
    :param intrinsic_matrix: 相机内参矩阵 (3x3)
    :param rgb_image: RGB图像 (H, W, 3)
    :param downsample_factor: 下采样因子
    :return: Open3D点云对象（相机坐标系）
    """
    height, width = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 创建网格坐标
    u = np.arange(0, width, downsample_factor)
    v = np.arange(0, height, downsample_factor)
    uu, vv = np.meshgrid(u, v)

    # 计算3D点 - OpenCV格式下的坐标系处理
    z = depth_map[vv, uu]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy  # OpenCV格式，Y轴无需取负

    # 创建点云
    points_c = np.stack((x, y, z), axis=-1)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # 过滤无效点（深度为0）
    valid_mask = (z.flatten() > 0) & (z.flatten() < 1)
    points = points[valid_mask]

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 添加颜色信息
    if rgb_image is not None:
        colors = rgb_image[vv, uu].reshape(-1, 3)[valid_mask] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        np.save(os.path.join(PROJECT_ROOT, "data", "color_valid.npy"), colors)
        print("形状 }", colors.shape)

    return pcd,points_c

# 获取世界坐标系的点云

def point_cloud_world(depth_map, intrinsic_matrix, extrinsic_matrix=None, rgb_image=None, downsample_factor=2):
    """
    将深度图转换为点云（修正坐标系方向）
    :param depth_map: 深度图 (H, W)
    :param intrinsic_matrix: 相机内参矩阵 (3x3)
    :param extrinsic_matrix: 相机外参矩阵 (4x4)
    :param rgb_image: RGB图像 (H, W, 3)
    :param downsample_factor: 下采样因子
    :return: Open3D点云对象和世界坐标系的点云数组
    """
    height, width = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 创建网格坐标
    u = np.arange(0, width, downsample_factor)
    v = np.arange(0, height, downsample_factor)
    uu, vv = np.meshgrid(u, v)

    # 计算3D点 - OpenCV格式下的坐标系处理
    z = depth_map[vv, uu]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy  # OpenCV格式，Y轴无需取负

    # 创建点云数组（相机坐标系）
    points_c = np.stack((x, y, z), axis=-1)  # 形状 (H', W', 3)
    
    # 创建扁平化的点云（用于Open3D）
    points_flat = points_c.reshape(-1, 3)
    
    # 过滤无效点（深度为0）
    valid_mask = (z.flatten() > 0) & (z.flatten() < 1)
    points_valid = points_flat[valid_mask]

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_valid)

    # 添加颜色信息
    
    if rgb_image is not None:
        colors = rgb_image[vv, uu].reshape(-1, 3)[valid_mask] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        

    # 将点云转换到世界坐标系
    if extrinsic_matrix is not None:
        # 转换Open3D点云
        pcd.transform(extrinsic_matrix)  # 相机坐标系 → 世界坐标系
        
        # 转换points_c到世界坐标系（保持原始网格结构）
        # 1. 转换为齐次坐标
        points_c_homogeneous = np.concatenate([
            points_c, 
            np.ones((*points_c.shape[:-1], 1))  # 添加齐次坐标分量
        ], axis=-1)
        
        # 2. 应用外参变换
        points_c_world = np.einsum('ij,...j->...i', extrinsic_matrix, points_c_homogeneous)
        
        # 3. 转换回3D坐标（移除齐次坐标）
        points_c_world = points_c_world[..., :3]
    else:
        points_c_world = points_c  # 如果没有外参矩阵，保持相机坐标系
    return pcd, points_c_world