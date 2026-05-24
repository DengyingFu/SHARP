"""
SmolVLM Model Service 客户端
用于调用模型服务的客户端库
"""

import requests
import base64
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from io import BytesIO


class SmolVLMClient:
    """SmolVLM模型服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        """
        初始化客户端
        
        参数:
        - base_url: 服务地址，默认为 http://localhost:1234
        """
        self.base_url = base_url.rstrip('/')
        connect_timeout = float(os.getenv("SMOL_CONNECT_TIMEOUT", "5"))
        read_timeout = float(os.getenv("SMOL_READ_TIMEOUT", "90"))
        self.timeout = (connect_timeout, read_timeout)
    
    def health_check(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"健康检查失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def infer_with_url(
        self,
        prompt: str,
        image_url: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        scene_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用图像URL进行推理
        
        参数:
        - prompt: 文本提示
        - image_url: 图像URL地址
        - system_prompt: 系统提示
        - max_tokens: 最大生成token数
        - scene_graph: 场景图信息(JSON字符串)
        
        返回:
        - 推理结果字典，包含 status, result, error
        """
        payload = {
            "prompt": prompt,
            "image_url": image_url,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "scene_graph": scene_graph
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/infer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"推理请求失败: {e}")
            return {"status": "error", "result": "", "error": str(e)}
    
    def infer_with_base64(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        scene_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用Base64编码的图像进行推理
        
        参数:
        - prompt: 文本提示
        - image_base64: Base64编码的图像数据
        - system_prompt: 系统提示
        - max_tokens: 最大生成token数
        - scene_graph: 场景图信息(JSON字符串)
        
        返回:
        - 推理结果字典
        """
        payload = {
            "prompt": prompt,
            "image_base64": image_base64,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "scene_graph": scene_graph
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/infer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"推理请求失败: {e}")
            return {"status": "error", "result": "", "error": str(e)}
    
    def get_smolvlm_response(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        scene_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用本地图像文件进行推理
        
        参数:
        - prompt: 文本提示
        - image_path: 本地图像文件路径
        - system_prompt: 系统提示
        - max_tokens: 最大生成token数
        - scene_graph: 场景图信息(JSON字符串)
        
        返回:
        - 推理结果字典
        """
        try:
            # 读取图像文件并转换为Base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            result = self.infer_with_base64(
                prompt,
                image_base64,
                system_prompt,
                max_tokens,
                scene_graph
            )
            if not isinstance(result, dict):
                return ""
            if result.get("status") == "error":
                return f"Error: {result.get('error', 'smolvlm infer failed')}"
            pure_answer = str(result.get('result', '')).split("Assistant:")[-1].strip()
            return pure_answer
        except Exception as e:
            print(f"读取图像失败: {e}")
            return f"Error: {str(e)}"
    
    def infer_with_file_upload(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        scene_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用文件上传的方式进行推理
        
        参数:
        - prompt: 文本提示
        - image_path: 本地图像文件路径
        - system_prompt: 系统提示
        - max_tokens: 最大生成token数
        - scene_graph: 场景图信息(JSON字符串)
        
        返回:
        - 推理结果字典
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'prompt': prompt,
                    'system_prompt': system_prompt,
                    'max_tokens': max_tokens,
                    'scene_graph': scene_graph
                }
                response = requests.post(
                    f"{self.base_url}/infer_with_file",
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"文件上传推理失败: {e}")
            return {"status": "error", "result": "", "error": str(e)}

    def infer_text_only(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        scene_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        仅使用文本进行推理

        参数:
        - prompt: 文本提示
        - system_prompt: 系统提示
        - max_tokens: 最大生成token数
        - scene_graph: 场景图信息(JSON字符串)

        返回:
        - 推理结果字典
        """
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "scene_graph": scene_graph
        }

        try:
            response = requests.post(
                f"{self.base_url}/infer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"推理请求失败: {e}")
            return {"status": "error", "result": "", "error": str(e)}
    
    def get_config(self) -> Dict[str, Any]:
        """获取服务配置"""
        try:
            response = requests.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取配置失败: {e}")
            return {"error": str(e)}
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """更新服务配置"""
        try:
            response = requests.post(
                f"{self.base_url}/config",
                json=config,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"更新配置失败: {e}")
            return {"status": "error", "error": str(e)}

import os
def process_all_images_in_folder(image_folder, output_folder):
    """
    遍历文件夹下所有图片，推理并保存结果到txt
    :param image_folder: 图片文件夹路径
    :param output_folder: 结果txt保存路径
    """
    # 1. 创建输出文件夹（若不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. 定义支持的图片格式（可根据需要扩展）
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    i=0
    # 3. 遍历图片文件夹
    for file_name in os.listdir(image_folder):
        if i>10:
            break
        i += 1
        # 过滤非图片文件
        if not file_name.lower().endswith(supported_formats):
            continue
        
        # 拼接图片完整路径
        image_path = os.path.join(image_folder, file_name)
        print(f"\n=== 处理图片: {image_path} ===")
        
        try:
            # 4. 调用推理接口（保留你的原始prompt逻辑）
            result = client.get_smolvlm_response(
                prompt="""Describe all objects in the image.""",
                image_path=image_path,
                max_tokens=500
            )
            # print(f"推理结果: {result[:50]}...")  # 仅打印前50字符，避免过长
            
            # 5. 生成txt文件名（和图片同名，后缀改为txt）
            txt_file_name = os.path.splitext(file_name)[0] + '.txt'
            txt_file_path = os.path.join(output_folder, txt_file_name)
            
            # 6. 保存结果到txt文件
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(f"图片路径: {image_path}\n")  # 可选：保存图片路径
                f.write(f"推理结果:\n{result}\n")
            
            print(f"结果已保存到: {txt_file_path}")
        
        except Exception as e:
            # 容错：单个图片处理失败不影响整体流程
            print(f"处理图片 {file_name} 失败: {str(e)}")
            # 可选：保存错误信息到txt
            error_txt_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '_error.txt')
            with open(error_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"图片路径: {image_path}\n")
                f.write(f"错误信息: {str(e)}\n")
# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = SmolVLMClient("http://localhost:1234")
    # 配置路径（根据你的实际路径修改）
    IMAGE_FOLDER = "/home/zty_group/fdy/data/collect_1215/"  # 图片所在文件夹
    OUTPUT_FOLDER = "/home/zty_group/fdy/data/collect_1215/recg"  # 结果保存文件夹
    
    # 1. 检查服务健康状态
    print("=== 健康检查 ===")
    health = client.health_check()
    print(health)
     # 执行批量处理
    # print(f"开始处理文件夹: {IMAGE_FOLDER}")
    # process_all_images_in_folder(IMAGE_FOLDER, OUTPUT_FOLDER)
    # print("\n=== 所有图片处理完成 ===")
    # # 3. 使用场景图和系统提示进行推理
    print("\n=== 使用场景图和系统提示进行推理 ===")
    result = client.get_smolvlm_response(
        prompt="""Describe all objects in the image.""",
        image_path="/home/zty_group/fdy/smolVLM/4.png",
        # system_prompt="",
        # scene_graph="Scene Graph"+scene_graph,
        max_tokens=500
    )
    print(f"结果: {result}")