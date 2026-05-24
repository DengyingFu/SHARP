import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import base64
import requests
import yaml
import numpy as np
import random
from sklearn.cluster import DBSCAN
import open3d as o3d
import torch
import supervision as sv
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from sam.segmentation import sam, grounding_dino as detection
from tools.rgbdepth_to_pcd import point_cloud_camera, point_cloud_world
from openai import OpenAI
class OpenAIWrapper:
    """
    A wrapper for OpenAI-compatible APIs, configured for Qwen-VL via Alibaba Cloud Dashscope.
    """
    def __init__(self, config_path='ambiguity/configs/models.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.api_key = os.getenv("DASHSCOPE_API_KEY", config['openai']['api_key'])
        self.base_url = config['openai']['base_url']
        self.vl_model = config['openai']['qwen_vl_model']
        self.llm_model = config['openai']['qwen_llm_model']
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_vl_completion_stream(self, prompt,message, image_path):
        """
        Get a completion from the Vision-Language Model (Qwen-VL).

        Args:
            prompt (str): The text prompt.
            image_path (str): The path to the local image file.

        Returns:
            str: The content of the model's response.
        """
        # Function to encode the image to base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""     # 定义完整回复
        is_answering = False   # 判断是否结束思考过程并开始回复
        enable_thinking = False
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=self.vl_model,
            messages=[
                {
                    "role": "system",
                    "content":prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": message},
                    ],
                },
            ],
            stream=True,
            # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
            extra_body={
                'enable_thinking': True,
                "thinking_budget": 81920},

            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )

        if enable_thinking:
            print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content
        return answer_content
    
    def get_vl_completion(self, prompt,message, image_path):
        """
        Get a completion from the Vision-Language Model (Qwen-VL).

        Args:
            prompt (str): The text prompt.
            image_path (str): The path to the local image file.

        Returns:
            str: The content of the model's response.
        """
        # Function to encode the image to base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        base64_image = encode_image(image_path)
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        answer_content = ""     # 定义完整回复
        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=self.vl_model,
            messages=[
                {
                    "role": "system",
                    "content":prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": message},
                    ],
                },
            ],
            stream=False,
        )
        answer_content = completion.choices[0].message.content
        
        return answer_content
        
    def get_llm_completion_stream(self, messages):
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=self.llm_model,  # 您可以按需更换为其它深度思考模型
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True
        )
        is_answering = False  # 是否进入回复阶段
        full_response = ""  # 用于累积完整回复
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                    pass
            if hasattr(delta, "content") and delta.content:
                full_response += delta.content  # 累积内容
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)
        return full_response
    
    def get_llm_completion(self, messages):
        """
        Get a completion from the Large Language Model.

        Args:
            messages (list): A list of message objects (e.g., [{"role": "system", "content": "..."}, ...]).

        Returns:
            str: The content of the model's response.
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=self.llm_model,  
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=False,
            temperature=0
        )
        # 提取完整输出内容
        full_response = completion.choices[0].message.content
        return full_response




class DinoWrapper:
    """
    A placeholder wrapper for the DINO object detection model.
    """
    def __init__(self, config_path='../configs/models.yaml'):
        # In a real implementation, you would load the model here.
        print("DINO model wrapper initialized (placeholder).")
        self.Loading_Model()

    def remove_outliers(self,points):
        if len(points) > 100000:
            indices = random.sample(range(len(points)), 100000)
            points = points[indices]

        xyzs = points[:, :3]

        centroid = xyzs.mean(axis=0)
        distances_to_centroid = np.linalg.norm(xyzs - centroid, axis=1)
        weights = 1 + distances_to_centroid / distances_to_centroid.max()
        weighted_xyzs = xyzs * weights[:, np.newaxis]

        # DBSCAN
        try:
            clustering = DBSCAN(eps=0.05, min_samples=6, n_jobs=-1).fit(weighted_xyzs)
        except ValueError:
            print("DBSCAN failed, return all points")
            return points

        labels = clustering.labels_
        total_points = len(labels)
        num_clusters = labels.max() + 1
        cluster_sizes = [(i, np.sum(labels == i)) for i in range(num_clusters)]
        threshold = total_points * 0.05
        valid_clusters = [i for i, size in cluster_sizes if size > threshold]

        indices = np.where(np.isin(labels, valid_clusters))[0]
        filtered_points = points[indices]

        if len(filtered_points) < 0.8 * len(points):
            filtered_points = points

        return filtered_points

    def get_scene_graph(self, image, pcd, mask, object_names, output_folder="./"):
        if len(mask) == 0:
            return [], []
        n, h, w = mask.shape
        image = np.array(image)

        objects_info = []
        objects_dict = []
        for i in range(n):
            object_mask = mask[i]
            segmented_object = pcd[object_mask]  #得到目标物体的点云（无色）
            segmented_image = image[object_mask] #得到目标物体的分割图
            colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
            np.save(os.path.join(output_folder, f"obj_{i + 1}.npy"), colored_object_pcd)

            # segmented_object = self.remove_outliers(segmented_object) #过滤一下
            # colored_object_pcd = self.remove_outliers(colored_object_pcd)

            #得到目标物的坐标信息，单位是m（世界坐标系下，点云是世界坐标系下的）
            min_values = np.round(segmented_object.min(axis=0), 2).tolist()
            max_values = np.round(segmented_object.max(axis=0), 2).tolist()
            mean_values = np.round(segmented_object.mean(axis=0), 2).tolist()

            center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
            bbox = {
                "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
                "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
                "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
            }
            node = {
                'id': i + 1,
                'object name': object_names[i],
                'center': center,
                'bounding box': bbox
            }
            objects_info.append(node)

            node_dict = {
            "id": i + 1,
            "name": object_names[i],
            "center": mean_values,
            "bbox": {
                "min": min_values,
                "max": max_values
            }
            }
            objects_dict.append(node_dict)

        return objects_info, objects_dict

    def Loading_Model(self):
        self.detection_model = detection.get_model()
        self.sam_model = sam.get_model()

    def get_object_localizations(self, image, object_nouns, results_dir):
        """
        Placeholder for getting object localizations using DINO.

        Args:
            image: The input image (e.g., a PIL image or numpy array).
            object_nouns (list): A list of object nouns to detect (e.g., ["apple", "bowl"]).

        Returns:
            list: A list of dictionaries, each containing the object text and its bounding box.
        """
        print(f"DINO: Pretending to detect {object_nouns} in the image.")
        detections = detection.get_detections(image, object_nouns, self.detection_model, output_folder=results_dir,box_threshold=0.35, text_threshold=0.35, nms_threshold=0.8)
        print(detections)
        mask, ann_img, object_names = sam.get_mask(
        image, object_nouns, self.sam_model, detections, output_folder=results_dir,selected_indices=None)

        return detections, mask, object_names


class Owlv2Wrapper:
    """
    A wrapper for the Owlv2 object detection model using Hugging Face Transformers.
    """
    def __init__(self, config_path='ambiguity/configs/models.yaml'):
        print("Owlv2 model wrapper initialized.")
        owl_model_dir = os.path.join(project_root, "Owlv2")
        self.processor = Owlv2Processor.from_pretrained(owl_model_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(owl_model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.sam_model = sam.get_model()

    def remove_outliers(self,points):
        if len(points) > 100000:
            indices = random.sample(range(len(points)), 100000)
            points = points[indices]

        xyzs = points[:, :3]

        centroid = xyzs.mean(axis=0)
        distances_to_centroid = np.linalg.norm(xyzs - centroid, axis=1)
        weights = 1 + distances_to_centroid / distances_to_centroid.max()
        weighted_xyzs = xyzs * weights[:, np.newaxis]

        # DBSCAN
        try:
            clustering = DBSCAN(eps=0.05, min_samples=6, n_jobs=-1).fit(weighted_xyzs)
        except ValueError:
            print("DBSCAN failed, return all points")
            return points

        labels = clustering.labels_
        total_points = len(labels)
        num_clusters = labels.max() + 1
        cluster_sizes = [(i, np.sum(labels == i)) for i in range(num_clusters)]
        threshold = total_points * 0.05
        valid_clusters = [i for i, size in cluster_sizes if size > threshold]

        indices = np.where(np.isin(labels, valid_clusters))[0]
        filtered_points = points[indices]

        if len(filtered_points) < 0.8 * len(points):
            filtered_points = points

        return filtered_points

    def get_scene_graph(self, image, pcd, mask, object_names, output_folder="./"):
        if len(mask) == 0:
            return [], []
        n, h, w = mask.shape
        image = np.array(image)

        objects_info = []
        objects_dict = []
        for i in range(n):
            object_mask = mask[i]
            segmented_object = pcd[object_mask]  #得到目标物体的点云（无色）
            segmented_image = image[object_mask] #得到目标物体的分割图
            colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
            np.save(os.path.join(output_folder, f"obj_{i + 1}.npy"), colored_object_pcd)

            # segmented_object = self.remove_outliers(segmented_object) #过滤一下
            # colored_object_pcd = self.remove_outliers(colored_object_pcd)

            #得到目标物的坐标信息，单位是m（世界坐标系下，点云是世界坐标系下的）
            min_values = np.round(segmented_object.min(axis=0), 2).tolist()
            max_values = np.round(segmented_object.max(axis=0), 2).tolist()
            mean_values = np.round(segmented_object.mean(axis=0), 2).tolist()

            center = f"x: {mean_values[0]:.2f}, y: {mean_values[1]:.2f}, z: {mean_values[2]:.2f}"
            bbox = {
                "x_min ~ x_max": f"{min_values[0]:.2f} ~ {max_values[0]:.2f}",
                "y_min ~ y_max": f"{min_values[1]:.2f} ~ {max_values[1]:.2f}",
                "z_min ~ z_max": f"{min_values[2]:.2f} ~ {max_values[2]:.2f}"
            }
            node = {
                'id': i + 1,
                'object name': object_names[i],
                'center': center,
                'bounding box': bbox
            }
            objects_info.append(node)

            node_dict = {
            "id": i + 1,
            "name": object_names[i],
            "center": mean_values,
            "bbox": {
                "min": min_values,
                "max": max_values
            }
            }
            objects_dict.append(node_dict)

        return objects_info, objects_dict

    def get_object_localizations(self, image, object_nouns, results_dir):
        print(f"Owlv2: Detecting {object_nouns} in the image.")
        
        texts = [object_nouns]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        # image.size is (width, height), so reverse for (height, width)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        # Retrieve predictions for the first image
        i = 0
        result = results[i]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        
        print(f"Owlv2 found {len(boxes)} detections.")
        
        # Construct supervision Detections object
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=labels
        )
        
        if len(detections.xyxy) == 0:
             print("No objects detected via Owlv2.")
             # We might want to handle this gracefully for SAM, but SAM get_mask probably handles empty detections (or fails if accessing xyxy[0]).
             # Let's let it run as is, or maybe check before calling SAM?
             # DinoWrapper.get_detections handles empty by retrying with lower threshold. I'm not doing that here yet.

        # SAM Mask Generation
        mask, ann_img, object_names = sam.get_mask(
            image, object_nouns, self.sam_model, detections, output_folder=results_dir)

        return detections, mask, object_names


def generate_point_cloud(results_dir, last_hh, last_intrinsic, last_extrinsic, last_rgb):
    print("正在生成点云...")
    pcd_camera, points_c = point_cloud_camera(
        depth_map=last_hh,
        intrinsic_matrix=last_intrinsic,
        rgb_image=last_rgb,
        downsample_factor=1
    )
    pcd_world, points_w = point_cloud_world(
        depth_map=last_hh,
        intrinsic_matrix=last_intrinsic,
        extrinsic_matrix=last_extrinsic,
        rgb_image=last_rgb,
        downsample_factor=1
    )
    pcd_camera_path = os.path.join(results_dir, "point_camera_cloud.ply")
    o3d.io.write_point_cloud(pcd_camera_path, pcd_camera)
    pcd_path = os.path.join(results_dir, "point_world_cloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd_world)
    print(f"Saved point cloud to {pcd_path}")
    return pcd_camera, points_c, pcd_world, points_w

if __name__ == '__main__':
    # Example usage:
    # Make sure to create a dummy image file 'test_image.jpg' in the same directory
    # and a valid config file at '../configs/models.yaml'
    
    # Test Qwen-VL
    # vl_wrapper = OpenAIWrapper()
    # # Create a dummy image for testing if it doesn't exist
    # if not os.path.exists("test_image.jpg"):
    #     try:
    #         from PIL import Image
    #         dummy_img = Image.new('RGB', (200, 200), color = 'red')
    #         dummy_img.save("test_image.jpg")
    #         print("Created a dummy test_image.jpg.")
    #     except ImportError:
    #         print("PIL not installed. Cannot create a dummy image. Please provide a 'test_image.jpg'.")

    # if os.path.exists("test_image.jpg"):
    #     vl_prompt = "What is in this image?"
    #     vl_response = vl_wrapper.get_vl_completion(vl_prompt, "test_image.jpg")
    #     print("\n--- Qwen-VL Test ---")
    #     print(f"Prompt: {vl_prompt}")
    #     print(f"Response: {vl_response}")

    # # Test LLM
    # llm_wrapper = OpenAIWrapper()
    # llm_messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello, what is the capital of France?"}
    # ]
    # llm_response = llm_wrapper.get_llm_completion(llm_messages)
    # print("\n--- LLM Test ---")
    # print(f"Messages: {llm_messages}")
    # print(f"Response: {llm_response}")

    # Test DINO
    import cv2
    dino = DinoWrapper()
    image = cv2.imread(os.path.join(project_root, "data", "type1_benchmark_dataset_20260119_132346", "scene_5", "final_rgb.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dino_response = dino.get_object_localizations(image, ["banana",], project_root)
    print("\n--- DINO Test ---")
    print(f"Response: {dino_response}")
