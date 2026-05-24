#!/usr/bin/env python3
"""
Real World Grasp Execution Pipeline
整合相机采集、抓取检测、场景图生成、指令处理的完整流程

Author: dyfu
Date: 2026-02-03
"""

import os
import sys
import json
import time
import base64
import requests
import argparse
import numpy as np
import open3d as o3d
import cv2
from PIL import Image
from typing import Optional, Dict, List, Any
import scipy.io as scio
import yaml
from scipy.spatial.transform import Rotation as R

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
graspnet_dir = os.path.join(current_dir, 'src/GraspNet-CUDA12.4')
if graspnet_dir not in sys.path:
    sys.path.insert(0, graspnet_dir)

# 导入 RealSense 相机模块
try:
    from realsense_capture import RealSenseCamera
except ImportError:
    print("❌ 无法导入 realsense_capture，请检查路径")
    sys.exit(1)

# 导入 AnyGrasp 检测器
try:
    from Use_anyGrasp import GraspDetector
    from graspnetAPI import GraspGroup
except ImportError:
    print("❌ 无法导入 GraspDetector，请检查路径")
    sys.exit(1)


class RealWorldGraspPipeline:
    """真实场景抓取流程集成类"""
    
    def __init__(
        self,
        results_dir: str = "./data/real_world_results",
        perception_api: str = "http://10.105.0.3:3654/analyze_upload",
        language_api: str = "http://10.105.0.3:7412/process_command",
        anygrasp_api: str = "http://10.105.0.3:1596/grasp",
        camera_width: int = 1280,
        camera_height: int = 720,
        min_depth: float = 0.28,
        max_depth: float = 0.8,
        calibration_file: str = "/root/.ros/easy_handeye/jetcobot_eob_calib_eye_on_base.yaml",
        camera_auto_exposure: bool = False,
        camera_exposure: int = 5000,
        camera_gain: int = 64,
    ):
        """
        初始化真实场景抓取流程
        
        Args:
            results_dir: 结果保存目录
            perception_api: 场景图生成API地址
            language_api: 指令处理API地址
            anygrasp_api: AnyGrasp抓取检测API地址
            camera_width: 相机宽度
            camera_height: 相机高度
            min_depth: 最小深度（米）
            max_depth: 最大深度（米）
            calibration_file: 手眼标定文件路径
            camera_auto_exposure: 是否启用自动曝光（默认False）
            camera_exposure: 手动曝光时间（微秒，100-10000，默认5000）
            camera_gain: 手动增益值（0-128，默认64）
        """
        self.results_dir = results_dir
        self.perception_api = perception_api
        self.language_api = language_api
        self.anygrasp_api = anygrasp_api
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.calibration_file = calibration_file
        self.camera_auto_exposure = camera_auto_exposure
        self.camera_exposure = camera_exposure
        self.camera_gain = camera_gain
        self.R_base_camera = np.eye(3)
        self.t_base_camera = np.zeros(3)
        self._load_calibration()
        
        # 创建结果目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(results_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"✅ 初始化完成，结果保存至: {self.session_dir}")
        
        # 数据存储
        self.color_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.intrinsic_matrix: Optional[np.ndarray] = None
        self.depth_scale: Optional[float] = None
        self.workspace_mask: Optional[np.ndarray] = None
        self.scene_graph: Optional[Dict] = None
        self.grasp_group: Optional[GraspGroup] = None
        self.best_grasp: Optional[GraspGroup] = None
        self.points_cloud: Optional[np.ndarray] = None
        self.colors_cloud: Optional[np.ndarray] = None
        self.cloud: Optional[np.ndarray] = None

    def _load_calibration(self):
        """加载手眼标定（相机->基座），仅需旋转用于姿态对齐。"""
        try:
            if not os.path.exists(self.calibration_file):
                print(f"⚠️ 未找到手眼标定文件，使用单位旋转: {self.calibration_file}")
                return

            with open(self.calibration_file, 'r') as f:
                calib_data = yaml.safe_load(f)

            tf = calib_data.get('transformation', {})
            translation = np.array([
                float(tf.get('x', 0.0)),
                float(tf.get('y', 0.0)),
                float(tf.get('z', 0.0)),
            ])
            quaternion = np.array([
                float(tf.get('qx', 0.0)),
                float(tf.get('qy', 0.0)),
                float(tf.get('qz', 0.0)),
                float(tf.get('qw', 1.0)),
            ])

            self.R_base_camera = R.from_quat(quaternion).as_matrix()
            self.t_base_camera = translation
            print("✅ 手眼标定加载成功 (Camera -> Base) 用于姿态对齐")
        except Exception as exc:
            print(f"⚠️ 手眼标定加载失败，使用单位旋转: {exc}")
            self.R_base_camera = np.eye(3)
            self.t_base_camera = np.zeros(3)
        
    def step1_capture_scene(self) -> bool:
        """
        步骤1: 采集场景图像
        Returns:
            bool: 是否成功采集
        """
        print("\n" + "="*60)
        print("步骤 1/6: 采集场景图像")
        print("="*60)
        
        try:
            # 初始化相机
            print("正在启动 RealSense 相机...")
            camera = RealSenseCamera(
                width=self.camera_width,
                height=self.camera_height,
                auto_exposure=self.camera_auto_exposure,
                exposure=self.camera_exposure if not self.camera_auto_exposure else None,
                gain=self.camera_gain if not self.camera_auto_exposure else None,
            )
            
            self.intrinsic_matrix = camera.intrinsics
            self.depth_scale = camera.depth_scale
            
            print("\n相机控制:")
            print("  按 's' 键保存当前帧")
            print("  按 'a' 键切换自动曝光")
            print("  按 '+' 键增加曝光（手动模式）")
            print("  按 '-' 键减少曝光（手动模式）")
            print("  按 ']' 键增加增益")
            print("  按 '[' 键减少增益")
            print("  按 'i' 键显示当前设置")
            print("  按 'q' 键退出")
            
            captured = False
            while not captured:
                # 采集帧
                color_image, depth_image = camera.capture_frame()
                
                if color_image is None or depth_image is None:
                    continue
                
                # 可视化
                camera.visualize(color_image, depth_image)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # 保存数据
                    self.color_image = color_image
                    self.depth_image = depth_image
                    self.workspace_mask = camera.generate_workspace_mask(
                        depth_image,
                        min_depth=self.min_depth,
                        max_depth=self.max_depth
                    )
                    
                    # 保存到目录
                    data_dir = os.path.join(self.session_dir, "captured_data")
                    camera.save_data(data_dir, color_image, depth_image, self.workspace_mask)
                    
                    print(f"✅ 场景数据已保存到: {data_dir}")
                    captured = True
                    
                elif key == ord('a'):
                    # 切换自动曝光
                    settings = camera.get_current_settings()
                    if settings:
                        camera.enable_auto_exposure(not settings['auto_exposure'])
                
                elif key == ord('+') or key == ord('='):
                    # 增加曝光
                    settings = camera.get_current_settings()
                    if settings and not settings['auto_exposure']:
                        new_exposure = min(settings['exposure'] + 500, 10000)
                        camera.set_exposure(new_exposure)
                    else:
                        print("请先关闭自动曝光（按 'a' 键）")
                
                elif key == ord('-') or key == ord('_'):
                    # 减少曝光
                    settings = camera.get_current_settings()
                    if settings and not settings['auto_exposure']:
                        new_exposure = max(settings['exposure'] - 500, 100)
                        camera.set_exposure(new_exposure)
                    else:
                        print("请先关闭自动曝光（按 'a' 键）")
                
                elif key == ord(']'):
                    # 增加增益
                    settings = camera.get_current_settings()
                    if settings:
                        new_gain = min(settings['gain'] + 8, 128)
                        camera.set_gain(new_gain)
                
                elif key == ord('['):
                    # 减少增益
                    settings = camera.get_current_settings()
                    if settings:
                        new_gain = max(settings['gain'] - 8, 0)
                        camera.set_gain(new_gain)
                
                elif key == ord('i'):
                    # 显示当前设置
                    settings = camera.get_current_settings()
                    if settings:
                        print(f"\n--- 当前设置 ---")
                        print(f"自动曝光: {'开启' if settings['auto_exposure'] else '关闭'}")
                        print(f"曝光: {settings['exposure']:.0f} 微秒")
                        print(f"增益: {settings['gain']:.0f}")
                        print(f"----------------\n")
                    
                elif key == ord('q'):
                    print("❌ 用户取消采集")
                    camera.stop()
                    return False
            
            camera.stop()
            return True
            
        except Exception as e:
            print(f"❌ 采集场景失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_detect_grasps(self, vis: bool = False) -> bool:
        """
        步骤2: 检测抓取姿势
        Returns:
            bool: 是否成功检测
        """
        print("\n" + "="*60)
        print("步骤 2/6: 检测抓取姿势")
        print("="*60)
        
        if self.color_image is None or self.depth_image is None:
            print("❌ 未找到采集的图像数据")
            return False
        
        try:
            # 初始化抓取检测器
            print("正在初始化 AnyGrasp 检测器...")
            detector = GraspDetector(
                num_point=100000,
                anygrasp_endpoint=self.anygrasp_api,
                cylinder_radius=0.05,
                max_grip_width=0.08,
                anygrasp_samples=1000,
                apply_object_mask=True,
                dense_grasp=True,
                collision_detection=True,
            )
            self.detector = detector
            
            # 加载数据
            data_dir = os.path.join(self.session_dir, "captured_data")
            print(f"正在加载数据: {data_dir}")
            points, colors, cloud, lims = detector.load_data(data_dir)
            print(lims)
            lims = [-0.2, 0.2, -0.3, 0.0, 0.3, 0.5]
            # 保存点云数据供后续使用
            self.points_cloud = points
            self.colors_cloud = colors
            self.cloud = cloud
            
            # 检测抓取
            print("正在调用 AnyGrasp API 检测抓取姿势...")
            gg = detector.detect_grasps(points, colors, lims)
            
            if len(gg) == 0:
                print("❌ 未检测到有效的抓取姿势")
                return False
            
            self.grasp_group = gg
            
            # 保存抓取结果
            grasp_save_path = os.path.join(self.session_dir, "detected_grasps.npy")
            detector.save_grasps(gg, grasp_save_path)
            
            print(f"✅ 检测到 {len(gg)} 个抓取姿势")
            print(f"✅ 抓取数据已保存到: {grasp_save_path}")

            if vis:
                detector.visualize_grasps(gg, cloud, save_path=os.path.join(self.session_dir, "detected_grasps_visualization.png"))
            
            return True
            
        except Exception as e:
            print(f"❌ 抓取检测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_generate_scene_graph(self) -> bool:
        """
        步骤3: 生成场景图
        Returns:
            bool: 是否成功生成
        """
        print("\n" + "="*60)
        print("步骤 3/6: 生成场景图")
        print("="*60)
        
        if self.color_image is None:
            print("❌ 未找到RGB图像")
            return False
        
        try:
            # 保存RGB图像
            image_path = os.path.join(self.session_dir, "scene_image.png")
            Image.fromarray(self.color_image).save(image_path)
            
            print(f"正在调用场景图生成API: {self.perception_api}")
            
            # 调用场景图API
            with open(image_path, "rb") as f:
                files = {"image": ("scene_image.png", f, "image/png")}
                data = {
                    "results_dir": self.session_dir,
                    "use_oracle": "false",
                    "output_format": "v2",
                    "return_masks": "true",  # 需要返回masks
                }
                resp = requests.post(self.perception_api, files=files, data=data, timeout=300)
            
            resp.raise_for_status()
            self.scene_graph = resp.json()
            
            # 保存场景图
            scene_graph_path = os.path.join(self.session_dir, "scene_graph.json")
            with open(scene_graph_path, "w", encoding="utf-8") as f:
                json.dump(self.scene_graph, f, indent=2, ensure_ascii=False)
            
            # 打印检测到的物体
            objects = self.scene_graph.get("objects", [])
            print(f"\n✅ 场景图生成成功，检测到 {len(objects)} 个物体:")
            for obj in objects:
                print(f"  ID {obj['id']}: {obj['name']} ({obj['category']})")
            
            print(f"✅ 场景图已保存到: {scene_graph_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 场景图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_process_instruction(self) -> Optional[Dict]:
        """
        步骤4: 处理用户指令
        Returns:
            Dict: 处理结果（包含目标物体信息）
        """
        print("\n" + "="*60)
        print("步骤 4/6: 处理用户指令")
        print("="*60)
        
        if self.scene_graph is None:
            print("❌ 未找到场景图数据")
            return None
        
        try:
            # 获取物体类别列表
            objects = self.scene_graph.get("objects", [])
            obj_list = list(set([obj['category'].lower() for obj in objects if 'category' in obj]))
            
            print(f"场景中的物体类别: {obj_list}")
            
            # 获取用户指令
            print("\n请输入抓取指令 (例如: Pick up the yellow hammer):")
            try:
                instruction = input("> ").strip()
            except EOFError:
                instruction = ""
            
            if not instruction:
                print("❌ 未输入指令")
                return None
            
            print(f"指令: {instruction}")
            
            # 编码图像
            image_path = os.path.join(self.session_dir, "scene_image.png")
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            # 构建请求
            payload = {
                "instruction": instruction,
                "scene_graph": self.scene_graph,
                "image_base64": image_base64,
                "obj_list": obj_list,
                "task_type": "pick"
            }
            
            # 调用语言处理API
            print(f"正在调用指令处理API: {self.language_api}")
            response = requests.post(self.language_api, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 检查是否需要澄清
            if result.get("status") == "ask":
                print("\n需要澄清:")
                intermediate = result.get("intermediate_results", {})
                ambiguity_res = intermediate.get("resolve_ambiguity_results", [])
                
                if ambiguity_res:
                    question = ambiguity_res[-1].get("question", "未知问题")
                    print(f"问题: {question}")
                    
                    # 获取用户回答
                    print("请输入您的回答:")
                    try:
                        answer = input("> ").strip()
                    except EOFError:
                        answer = ""
                    
                    if not answer:
                        print("❌ 未输入回答")
                        return None
                    
                    # 使用历史记录重新请求
                    history = intermediate.get("history")
                    payload["clarification_history"] = history
                    payload["clarification_answer"] = answer
                    
                    print("正在重新处理...")
                    response = requests.post(self.language_api, json=payload, timeout=300)
                    response.raise_for_status()
                    result = response.json()
            
            # 检查最终结果
            if result.get("status") != "ok":
                print(f"❌ 指令处理失败: {result}")
                return None
            
            final_results = result.get("results", [])
            if not final_results:
                print("❌ 未识别到目标物体")
                return None
            
            # 保存结果
            result_path = os.path.join(self.session_dir, "language_result.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            target = final_results[0]  # 取第一个目标
            print(f"\n✅ 识别目标物体:")
            print(f"  ID: {target['id']}")
            print(f"  类别: {target['category']}")
            print(f"  属性: {target.get('attributes', 'N/A')}")
            
            print(f"✅ 指令处理结果已保存到: {result_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ 指令处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step5_filter_grasps(self, language_result: Dict) -> Optional[Dict]:
        """
        步骤5: 根据目标物体筛选抓取姿势
        Args:
            language_result: 语言处理结果
        Returns:
            Dict: 最佳抓取姿势信息
        """
        print("\n" + "="*60)
        print("步骤 5/6: 筛选抓取姿势")
        print("="*60)
        
        if self.grasp_group is None or len(self.grasp_group) == 0:
            print("❌ 未找到抓取数据")
            return None
        
        if self.scene_graph is None:
            print("❌ 未找到场景图")
            return None
        
        try:
            # 获取目标物体ID
            final_results = language_result.get("results", [])
            if not final_results:
                print("❌ 无有效目标")
                return None
            
            target = final_results[0]
            target_id = target['id']
            target_category = target['category']
            
            print(f"目标物体: ID={target_id}, 类别={target_category}")
            
            # 获取目标物体的mask和bbox
            masks = self.scene_graph.get("masks")
            bboxes = self.scene_graph.get("bbox", [])
            
            if masks is None or len(masks) == 0:
                print("❌ 场景图中未包含mask信息")
                return None
            
            # masks和objects是对齐的，ID从1开始
            mask_index = target_id - 1
            if mask_index < 0 or mask_index >= len(masks):
                print(f"❌ 目标ID {target_id} 超出范围")
                return None
            
            target_mask = np.array(masks[mask_index], dtype=np.uint8)
            print(f"目标mask形状: {target_mask.shape}")
            
            # 获取目标bbox（用于计算3D中心）
            if mask_index < len(bboxes):
                target_bbox = bboxes[mask_index]
                print(f"目标2D bbox: {target_bbox}")
            
            # 计算目标物体的3D信息
            # 从depth_image生成3D点云
            h, w = self.depth_image.shape
            fx = self.intrinsic_matrix[0, 0]
            fy = self.intrinsic_matrix[1, 1]
            cx = self.intrinsic_matrix[0, 2]
            cy = self.intrinsic_matrix[1, 2]
            
            # 生成3D点云（相机坐标系）
            depth_m = self.depth_image * self.depth_scale
            
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            z = depth_m
            x = (x_coords - cx) * z / fx
            y = (y_coords - cy) * z / fy
            
            points_3d = np.stack([x, y, z], axis=-1)  # (H, W, 3)
            
            # Resize mask to match depth image size
            if target_mask.shape != (h, w):
                target_mask_resized = cv2.resize(target_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                target_mask_resized = target_mask
            
            # 提取目标区域的3D点
            valid_mask = (target_mask_resized > 0) & (depth_m > 0)
            target_points_3d = points_3d[valid_mask]
            
            if len(target_points_3d) == 0:
                print("❌ 目标区域没有有效深度点")
                return None
            
            # 计算3D边界框和中心
            min_coords = np.min(target_points_3d, axis=0)
            max_coords = np.max(target_points_3d, axis=0)
            center_3d = (min_coords + max_coords) / 2
            
            print(f"目标3D中心: {center_3d}")
            print(f"目标3D范围: min={min_coords}, max={max_coords}")
            
            # 筛选抓取姿势：严格筛选在目标mask内的最垂直抓取
            print(f"\n正在从 {len(self.grasp_group)} 个抓取中筛选...")
            
            # 世界/基座坐标系下的垂直方向（+Z）
            # AnyGrasp 抓取坐标系: +X 为接近方向, +Y 为右侧, +Z 为向上
            vertical_direction_world = np.array([0, 0, 1])
            
            # 第一步：严格筛选 - 只保留抓取中心在目标物体内的抓取
            candidate_grasps = []
            
            for i, grasp in enumerate(self.grasp_group):
                grasp_translation = grasp.translation  # 抓取中心在相机坐标系中的位置 (x, y, z)
                
                # 将3D点投影回2D像素坐标，检查是否在mask内
                grasp_x, grasp_y, grasp_z = grasp_translation
                
                if grasp_z <= 0:  # 深度无效
                    continue
                
                # 投影到像素坐标
                pixel_x = int(grasp_x * fx / grasp_z + cx)
                pixel_y = int(grasp_y * fy / grasp_z + cy)
                
                # 检查是否在图像范围内
                if pixel_x < 0 or pixel_x >= w or pixel_y < 0 or pixel_y >= h:
                    continue
                
                # 检查是否在目标mask内
                if target_mask_resized[pixel_y, pixel_x] == 0:
                    continue
                
                # 计算到目标中心的距离
                distance = np.linalg.norm(grasp_translation - center_3d)
                
                # 计算垂直度：先将抓取接近方向(+X)从相机系转换到基座/世界系，再与世界+Z对齐度比较
                # AnyGrasp: rotation_matrix 的第1列(索引0)是接近方向
                approach_direction_cam = grasp.rotation_matrix[:, 0]
                approach_direction_world = self.R_base_camera @ approach_direction_cam
                # 归一化以防数值误差
                approach_direction_world = approach_direction_world / (np.linalg.norm(approach_direction_world) + 1e-9)

                # 计算与垂直方向的余弦相似度（取绝对值，因为可以从上或从下抓取）
                cos_angle = np.abs(np.dot(approach_direction_world, vertical_direction_world))
                verticality = cos_angle  # 越接近1越垂直
                
                # 计算垂直角度（度）用于显示
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                
                candidate_grasps.append({
                    'grasp': grasp,
                    'distance': distance,
                    'verticality': verticality,
                    'angle_deg': angle_deg,
                    'score': grasp.score,
                    'index': i,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y
                })
            
            print(f"  找到 {len(candidate_grasps)} 个在目标物体内的抓取")
            
            # 第二步：从候选中选择最垂直的抓取
            best_grasp = None
            best_distance = float('inf')
            best_verticality = 0
            best_raw_score = 0
            best_angle_deg = 90
            
            if len(candidate_grasps) > 0:
                # 打印候选信息（前10个）
                print("\n  候选抓取信息（按垂直度排序，显示前10个）:")
                sorted_candidates = sorted(candidate_grasps, key=lambda x: x['verticality'], reverse=True)
                for i, cand in enumerate(sorted_candidates[:10]):
                    print(f"    [{i+1}] 垂直度:{cand['verticality']:.3f} (角度:{cand['angle_deg']:.1f}°) "
                          f"分数:{cand['score']:.3f} 距离:{cand['distance']:.3f}m "
                          f"位置:[{cand['pixel_x']}, {cand['pixel_y']}]")
                
                # 筛选策略：优先选择垂直度最高的，如果垂直度相近（差异<0.1），则选分数更高的
                min_angle = sorted_candidates[0]['angle_deg']
                threshold = 5  # 垂直度差异阈值
                
                # 收集垂直度接近最大值的候选
                top_vertical_candidates = [
                    c for c in sorted_candidates 
                    if c['angle_deg'] <= min_angle + threshold
                ]
                
                print(f"\n  筛选出 {len(top_vertical_candidates)} 个度数小的候选（度数 <= {min_angle + threshold:.3f}）")
                
                # 从高垂直度候选中选择分数最高的
                best_candidate = max(top_vertical_candidates, key=lambda x: x['score'])
                
                best_grasp = best_candidate['grasp']
                best_distance = best_candidate['distance']
                best_verticality = best_candidate['verticality']
                best_raw_score = best_candidate['score']
                best_angle_deg = best_candidate['angle_deg']
                
                print(f"\n  ✅ 最佳抓取:")
                print(f"    - 垂直度: {best_verticality:.4f} (与垂直方向夹角: {best_angle_deg:.1f}°)")
                print(f"    - 抓取分数: {best_raw_score:.4f}")
                print(f"    - 距离目标中心: {best_distance:.4f} m")
                print(f"    - 2D位置: [{best_candidate['pixel_x']}, {best_candidate['pixel_y']}]")
            
            # 第三步：如果严格筛选后没有候选，尝试放宽mask限制
            if best_grasp is None:
                print("\n  ⚠️ 目标mask内未找到抓取，尝试3D bbox范围...")
                
                # 使用3D bbox + 小容差
                margin = 0.02  # 2cm容差
                fallback_candidates = []
                
                for i, grasp in enumerate(self.grasp_group):
                    grasp_translation = grasp.translation
                    
                    # 检查是否在3D bbox范围内
                    in_bbox = np.all(grasp_translation >= (min_coords - margin)) and \
                              np.all(grasp_translation <= (max_coords + margin))
                    
                    if not in_bbox:
                        continue
                    
                    distance = np.linalg.norm(grasp_translation - center_3d)
                    approach_direction_cam = grasp.rotation_matrix[:, 0]
                    approach_direction_world = self.R_base_camera @ approach_direction_cam
                    approach_direction_world = approach_direction_world / (np.linalg.norm(approach_direction_world) + 1e-9)
                    cos_angle = np.abs(np.dot(approach_direction_world, vertical_direction_world))
                    verticality = cos_angle
                    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                    
                    fallback_candidates.append({
                        'grasp': grasp,
                        'distance': distance,
                        'verticality': verticality,
                        'angle_deg': angle_deg,
                        'score': grasp.score,
                    })
                
                if len(fallback_candidates) > 0:
                    print(f"  在3D bbox范围内找到 {len(fallback_candidates)} 个候选")
                    
                    # 同样优先选择最垂直的
                    sorted_fb = sorted(fallback_candidates, key=lambda x: x['verticality'], reverse=True)
                    max_vert = sorted_fb[0]['verticality']
                    top_vert = [c for c in sorted_fb if c['verticality'] >= max_vert - 0.1]
                    best_candidate = max(top_vert, key=lambda x: x['score'])
                    
                    best_grasp = best_candidate['grasp']
                    best_distance = best_candidate['distance']
                    best_verticality = best_candidate['verticality']
                    best_raw_score = best_candidate['score']
                    best_angle_deg = best_candidate['angle_deg']
                    
                    print(f"  ✅ 备选抓取: 垂直度={best_verticality:.3f}, 分数={best_raw_score:.3f}")
                else:
                    print("  ❌ 3D bbox范围内也未找到抓取")
                    return None
            self.best_grasp = GraspGroup(np.expand_dims(best_grasp.grasp_array, axis=0))
            
            print(f"✅ 找到最佳抓取姿势:")
            print(f"  距离目标中心: {best_distance:.4f} m")
            print(f"  抓取得分: {best_grasp.score:.4f}")
            print(f"  抓取位置: {best_grasp.translation}")
            print(f"  抓取宽度: {best_grasp.width:.4f} m")
            
            # 构建返回结果
            grasp_info = {
                "target_id": target_id,
                "target_category": target_category,
                "target_center": center_3d.tolist(),
                "grasp_score": float(best_grasp.score),
                "grasp_position": best_grasp.translation.tolist(),
                "grasp_rotation": best_grasp.rotation_matrix.tolist(),
                "grasp_width": float(best_grasp.width),
                "distance_to_target": float(best_distance),
            }
            
            # 保存结果
            grasp_info_path = os.path.join(self.session_dir, "filtered_grasp.json")
            with open(grasp_info_path, "w") as f:
                json.dump(grasp_info, f, indent=2)
            
            print(f"✅ 抓取信息已保存到: {grasp_info_path}")
            
            return grasp_info
            
        except Exception as e:
            print(f"❌ 筛选抓取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def step6_visualize_results(self, grasp_info: Dict) -> bool:
        """
        步骤6: 可视化结果
        Args:
            grasp_info: 抓取信息
        Returns:
            bool: 是否成功可视化
        """
        print("\n" + "="*60)
        print("步骤 6/6: 可视化结果")
        print("="*60)
        
        try:
            self.detector.visualize_grasps(self.best_grasp, self.cloud, save_path=os.path.join(self.session_dir, "best_grasp_visualization.png"))
            

            # 创建点云
            # data_dir = os.path.join(self.session_dir, "captured_data")
            
            # # 重新加载完整点云用于可视化
            # detector = GraspDetector(
            #     num_point=100000,
            #     anygrasp_endpoint=self.anygrasp_api,
            # )
            # _, _, full_cloud, _ = detector.load_data(data_dir)
            
            # # 创建抓取姿势的Open3D几何体
            # grasp_position = np.array(grasp_info["grasp_position"])
            # grasp_rotation = np.array(grasp_info["grasp_rotation"])
            
            # # 创建坐标系表示抓取姿势
            # grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            
            # # 应用旋转和平移
            # grasp_frame.rotate(grasp_rotation, center=[0, 0, 0])
            # grasp_frame.translate(grasp_position)
            
            # # 创建目标中心标记
            # target_center = np.array(grasp_info["target_center"])
            # target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            # target_sphere.translate(target_center)
            # target_sphere.paint_uniform_color([1, 0, 0])  # 红色
            
            # # 世界坐标系
            # world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
            
            # # 可视化
            # print("\n正在启动可视化窗口...")
            # print("控制说明:")
            # print("  鼠标左键: 旋转视角")
            # print("  鼠标滚轮: 缩放")
            # print("  按 'q' 键关闭窗口")
            
            # geometries = [full_cloud, grasp_frame, target_sphere, world_frame]
            
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(window_name='抓取可视化结果', width=1280, height=720)
            
            # for geometry in geometries:
            #     vis.add_geometry(geometry)
            
            # opt = vis.get_render_option()
            # opt.background_color = np.asarray([0.95, 0.95, 0.95])
            # opt.mesh_show_back_face = True
            # opt.point_size = 2.0
            
            # # 设置视角
            # ctr = vis.get_view_control()
            # ctr.set_front([0, 0, -1])
            # ctr.set_lookat(target_center)
            # ctr.set_up([0, -1, 0])
            # ctr.set_zoom(0.8)
            
            # vis.run()
            # vis.destroy_window()
            
            print("✅ 可视化完成")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """运行完整流程"""
        print("\n" + "="*60)
        print("真实场景抓取任务流程")
        print("="*60)
        print(f"结果保存目录: {self.session_dir}")
        print()
        
        # 步骤1: 采集场景
        if not self.step1_capture_scene():
            print("\n❌ 流程终止: 场景采集失败")
            return
        
        # 步骤2: 检测抓取
        if not self.step2_detect_grasps(vis=True):
            print("\n❌ 流程终止: 抓取检测失败")
            return
        
        # 步骤3: 生成场景图
        if not self.step3_generate_scene_graph():
            print("\n❌ 流程终止: 场景图生成失败")
            return
        
        # 步骤4: 处理指令
        language_result = self.step4_process_instruction()
        if language_result is None:
            print("\n❌ 流程终止: 指令处理失败")
            return
        
        # 步骤5: 筛选抓取
        grasp_info = self.step5_filter_grasps(language_result)
        if grasp_info is None:
            print("\n❌ 流程终止: 抓取筛选失败")
            return
        
        # 步骤6: 可视化
        self.step6_visualize_results(grasp_info)
        
        # 完成
        print("\n" + "="*60)
        print("✅ 流程完成!")
        print("="*60)
        print(f"所有结果已保存至: {self.session_dir}")
        print()


def main():
    parser = argparse.ArgumentParser(description='真实场景抓取任务流程')
    parser.add_argument('--results_dir', type=str, default='./data/real_world_results_demo',
                        help='结果保存根目录')
    parser.add_argument('--perception_api', type=str, default='http://10.105.1.3:3654/analyze_upload',
                        help='场景图生成API地址')
    parser.add_argument('--language_api', type=str, default='http://10.105.1.3:7412/process_command',
                        help='指令处理API地址')
    parser.add_argument('--anygrasp_api', type=str, default='http://10.105.1.3:1596/grasp',
                        help='AnyGrasp抓取检测API地址')
    parser.add_argument('--camera_width', type=int, default=1280,
                        help='相机宽度')
    parser.add_argument('--camera_height', type=int, default=720,
                        help='相机高度')
    parser.add_argument('--min_depth', type=float, default=0.3,
                        help='最小深度（米）')
    parser.add_argument('--max_depth', type=float, default=1.0,
                        help='最大深度（米）')
    parser.add_argument('--calibration', type=str, default='/root/.ros/easy_handeye/jetcobot_eob_calib_eye_on_base.yaml',
                        help='手眼标定文件（Camera->Base）路径，用于姿态对齐')
    parser.add_argument('--auto_exposure', action='store_true',
                        help='启用自动曝光（默认关闭，使用手动曝光）')
    parser.add_argument('--exposure', type=int, default=120,
                        help='手动曝光时间（微秒，范围100-10000，默认100）')
    parser.add_argument('--gain', type=int, default=64,
                        help='手动增益值（范围0-128，默认64）')
    
    args = parser.parse_args()
    
    # 创建流程实例
    pipeline = RealWorldGraspPipeline(
        results_dir=args.results_dir,
        perception_api=args.perception_api,
        language_api=args.language_api,
        anygrasp_api=args.anygrasp_api,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        calibration_file=args.calibration,
        camera_auto_exposure=args.auto_exposure,
        camera_exposure=args.exposure,
        camera_gain=args.gain,
    )
    
    # 运行流程
    pipeline.run()


if __name__ == "__main__":
    main()
