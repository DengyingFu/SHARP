import requests
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Any
import io
import json


class DinoServiceClient:
    """
    DINO 服务的客户端，用于方便地调用远程 DINO 服务进行对象检测和分割。
    
    使用示例:
        client = DinoServiceClient(base_url="http://localhost:8000")
        detections, mask, object_names = client.get_object_localizations(
            image_path="path/to/image.jpg",
            object_nouns=["apple", "banana", "cup"]
        )
    """
    
    def __init__(self, base_url: str = "http://localhost:5678"):
        """
        初始化 DINO 服务客户端。
        """
        self.base_url = base_url.rstrip('/')
        self.endpoint_localizations = f"{self.base_url}/get_object_localizations"
        self.endpoint_scene_graph = f"{self.base_url}/get_scene_graph"
    
    def get_object_localizations(
        self,
        image_path: str = None,
        image: Image.Image = None,
        image_array: np.ndarray = None,
        object_nouns: List[str] = None,
        results_dir: str=None
    ) -> Tuple[Any, np.ndarray, List[str]]:
        """
        调用 DINO 服务进行对象检测和分割。
        
        Args:
            image_path: 图像文件路径（三种输入方式之一）
            image: PIL Image 对象（三种输入方式之一）
            image_array: numpy 数组格式的图像（三种输入方式之一）
            object_nouns: 要检测的对象名称列表
            
        Returns:
            detections: 检测结果信息
            mask: 分割掩码（numpy 数组）
            object_names: 检测到的对象名称列表
            
        Raises:
            ValueError: 如果没有提供图像输入或对象名称列表
            requests.RequestException: 如果请求失败
        """
        if object_nouns is None or len(object_nouns) == 0:
            raise ValueError("必须提供至少一个对象名称")
        
        # 处理图像输入
        pil_image = None
        if image_path is not None:
            pil_image = Image.open(image_path).convert("RGB")
        elif image is not None:
            pil_image = image.convert("RGB")
        elif image_array is not None:
            pil_image = Image.fromarray(image_array).convert("RGB")
        else:
            raise ValueError("必须提供图像输入（image_path, image 或 image_array）")
        
        # 将图像转换为字节流
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # 准备请求数据
        files = {
            'image': ('image.png', img_byte_arr, 'image/png')
        }
        data = {
            'object_nouns': ','.join(object_nouns),
            'results_dir':results_dir
        }
        
        # 发送请求
        try:
            response = requests.post(
                self.endpoint_localizations,
                files=files,
                data=data,
                timeout=1000
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            detections = result.get('detections')
            mask = np.array(result.get('mask'))
            object_names = result.get('object_names', [])
            
            return detections, mask, object_names
            
        except requests.RequestException as e:
            raise Exception(f"调用 DINO 服务失败: {str(e)}")
    
    def get_scene_graph(
        self,
        image: Image.Image,
        pcd: np.ndarray,
        mask: np.ndarray,
        object_names: List[str],
        output_folder
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        调用 DINO 服务生成场景图。

        Args:
            image: PIL Image 对象。
            pcd: numpy 数组格式的点云数据。
            mask: numpy 数组格式的分割掩码。
            object_names: 检测到的对象名称列表。

        Returns:
            objects_info: 每个对象的信息。
            objects_dict: 对象的字典表示。

        Raises:
            ValueError: 如果输入无效。
            Exception: 如果请求失败。
        """
        if image is None or pcd is None or mask is None or not object_names:
            raise ValueError("必须提供图像、点云、掩码和对象名称。")

        # 将图像转换为字节流
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # 将 numpy 数组转换为 JSON 字符串，然后转换为字节流
        mask_json_str = json.dumps(mask.tolist())
        pcd_json_str = json.dumps(pcd.tolist())
        
        files = {
            'image': ('image.png', img_byte_arr, 'image/png'),
            'mask': ('mask.json', mask_json_str, 'application/json'),
            'pcd': ('pcd.json', pcd_json_str, 'application/json'),
        }
        data = {
            'object_names': ','.join(object_names),
            'output_folder':output_folder
        }
        
        try:
            response = requests.post(
                self.endpoint_scene_graph,
                files=files,
                data=data,
                timeout=1000  # Increased timeout for potentially larger data
            )
            response.raise_for_status()
            
            result = response.json()
            objects_info = result.get('objects_info', [])
            objects_dict = result.get('objects_dict', [])
            
            return objects_info, objects_dict
            
        except requests.RequestException as e:
            raise Exception(f"调用场景图服务失败: {str(e)}")

    def check_service_health(self) -> bool:
        """
        检查 DINO 服务是否可用。
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=200)
            return response.status_code == 200
        except requests.RequestException:
            return False


# 便捷函数
def get_object_localizations(
    image_path: str = None,
    image: Image.Image = None,
    image_array: np.ndarray = None,
    object_nouns: List[str] = None,
    service_url: str = "http://localhost:5678"
) -> Tuple[Any, np.ndarray, List[str]]:
    """
    便捷函数：直接调用 DINO 服务进行对象检测和分割。
    
    Args:
        image_path: 图像文件路径
        image: PIL Image 对象
        image_array: numpy 数组格式的图像
        object_nouns: 要检测的对象名称列表
        service_url: DINO 服务的 URL
        
    Returns:
        detections: 检测结果信息
        mask: 分割掩码（numpy 数组）
        object_names: 检测到的对象名称列表
    """
    client = DinoServiceClient(base_url=service_url)
    return client.get_object_localizations(
        image_path=image_path,
        image=image,
        image_array=image_array,
        object_nouns=object_nouns
    )


if __name__ == "__main__":
    client = DinoServiceClient()
    detections, mask, object_names = client.get_object_localizations(
    image_path='/home/zty_group/fdy/data/real_world_results_demo/session_20260207_043911/uploaded_image.png',
    object_nouns=['strawberry', 'spoon','screwdriver','banana','hammer'],
    results_dir='/home/zty_group/fdy/data/demo/'
)