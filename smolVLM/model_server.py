"""
SmolVLM Model Service
提供FastAPI服务来加载和推理SmolVLM模型
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import logging
import base64
from io import BytesIO
from contextlib import asynccontextmanager, nullcontext
from typing import Optional
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量存储模型和处理器
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
_request_count = 0

class ModelConfig:
    """模型配置"""
    MODEL_PATH = "weights/smolvlm-real"
    LORA_PATH = ""
    MAX_TOKENS = 500
    HOST = "0.0.0.0"
    PORT = 1234
    EMPTY_CACHE_EVERY_N_REQUESTS = 0
    USE_AUTOCAST = True
    
    @classmethod
    def update_from_dict(cls, config_dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


class InferenceRequest(BaseModel):
    """推理请求"""
    prompt: str
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    max_tokens: Optional[int] = 500
    scene_graph: Optional[str] = None


class InferenceResponse(BaseModel):
    """推理响应"""
    status: str
    result: str
    error: Optional[str] = None


def _maybe_release_cuda_cache():
    global _request_count
    if not torch.cuda.is_available():
        return
    interval = max(int(ModelConfig.EMPTY_CACHE_EVERY_N_REQUESTS or 0), 0)
    if interval <= 0:
        return
    _request_count += 1
    if _request_count % interval == 0:
        torch.cuda.empty_cache()


def _autocast_context():
    if device == "cuda" and bool(ModelConfig.USE_AUTOCAST):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def initialize_model():
    """初始化模型"""
    global model, processor, device
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        # 加载处理器
        processor_path = getattr(ModelConfig, "PROCESSOR_PATH", ModelConfig.MODEL_PATH)
        logger.info(f"从 {processor_path} 加载处理器...")
        processor = AutoProcessor.from_pretrained(processor_path)
        
        # 加载模型
        logger.info(f"从 {ModelConfig.MODEL_PATH} 加载模型...")
        model = AutoModelForVision2Seq.from_pretrained(
            ModelConfig.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)
        if ModelConfig.LORA_PATH:
            model = PeftModel.from_pretrained(model, ModelConfig.LORA_PATH)
        # 设置评估模式
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True
        
        logger.info("模型初始化成功!")
        return True
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        return False


def load_image_from_source(image_url: Optional[str] = None, image_base64: Optional[str] = None) -> Optional[Image.Image]:
    """从URL或Base64加载图像"""
    try:
        if image_url:
            image = load_image(image_url)
            return image.convert("RGB") if hasattr(image, "convert") else image
        elif image_base64:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        else:
            return None
    except Exception as e:
        logger.error(f"加载图像失败: {str(e)}")
        return None


def prepare_prompt_with_scene_graph(prompt: str, scene_graph: Optional[str] = None) -> str:
    """准备包含场景图的提示"""
    if scene_graph:
        return f"场景图信息:\n{scene_graph}\n\n用户提示: {prompt}"
    return prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化模型
    logger.info("启动应用，初始化模型...")
    if not initialize_model():
        logger.error("模型初始化失败，应用启动失败")
        raise RuntimeError("模型初始化失败")
    
    yield
    
    # 关闭时清理资源
    logger.info("应用关闭，清理资源...")
    # 可以在这里添加清理代码


# 创建FastAPI应用
app = FastAPI(
    title="SmolVLM Model Service",
    description="提供SmolVLM视觉语言模型的推理服务",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """健康检查"""
    allocated_gb = 0.0
    reserved_gb = 0.0
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3

    return {
        "status": "ok",
        "device": device,
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "cuda_allocated_gb": round(allocated_gb, 3),
        "cuda_reserved_gb": round(reserved_gb, 3),
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    执行推理
    """
    # 局部变量初始化，确保 finally 中能访问
    inputs = None
    generated_ids = None
    generated_texts = None
    
    try:
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="模型未初始化")
        
        # 验证输入
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt不能为空")
        
        # 加载图像
        image = load_image_from_source(
            request.image_url, 
            request.image_base64
        )
        
        # 准备提示
        final_prompt = prepare_prompt_with_scene_graph(
            request.prompt,
            request.scene_graph
        )
        
        # 创建消息
        messages = []
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        user_message = {
            "role": "user",
            "content": []
        }
        
        # 如果有图像，添加到消息中
        if image:
            user_message["content"].append({"type": "image"})
        
        user_message["content"].append({
            "type": "text",
            "text": final_prompt
        })
        messages.append(user_message)
        
        # 处理输入
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        if image:
            inputs = processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt"
            )
        else:
            inputs = processor(
                text=prompt_text,
                return_tensors="pt"
            )
        
        inputs = inputs.to(device)
        
        # 生成输出 - 使用 torch.no_grad() 替代 inference_mode，并显式管理
        with torch.no_grad():
            if str(device) == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=request.max_tokens or ModelConfig.MAX_TOKENS
                    )
            else:
                 generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or ModelConfig.MAX_TOKENS
                )
        
        # 解码结果
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        result = generated_texts[0] if generated_texts else ""
        
        logger.info(f"推理成功: {result[:100]}...")
        
        return InferenceResponse(
            status="success",
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"推理失败: {str(e)}", exc_info=True)
        return InferenceResponse(
            status="error",
            result="",
            error=str(e)
        )
    finally:
        # 显式清理资源
        if inputs is not None:
            del inputs
        if generated_ids is not None:
            del generated_ids
        if generated_texts is not None:
            del generated_texts
        if 'image' in locals() and image is not None:
            del image
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/infer_with_file", response_model=InferenceResponse)
async def infer_with_file(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    system_prompt: Optional[str] = Form(None),
    max_tokens: int = Form(500),
    scene_graph: Optional[str] = Form(None)
):
    """
    使用文件上传的方式执行推理
    """
    # 局部变量初始化
    inputs = None
    generated_ids = None
    generated_texts = None
    image = None

    try:
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="模型未初始化")
        
        # 读取上传的文件
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # 准备提示
        final_prompt = prepare_prompt_with_scene_graph(prompt, scene_graph)
        
        # 创建消息
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": final_prompt}
            ]
        })
        
        # 处理输入
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        
        # 生成输出 - 使用 torch.no_grad 和 autocast，与 infer 一致
        with torch.no_grad():
            if str(device) == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens
                    )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens
                )
        
        # 解码结果
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        result = generated_texts[0] if generated_texts else ""
        
        logger.info(f"文件推理成功: {result[:100]}...")
        
        return InferenceResponse(
            status="success",
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件推理失败: {str(e)}", exc_info=True)
        return InferenceResponse(
            status="error",
            result="",
            error=str(e)
        )
    finally:
        # 显式清理
        if inputs is not None:
            del inputs
        if generated_ids is not None:
            del generated_ids
        if generated_texts is not None:
            del generated_texts
        if image is not None:
            del image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "model_path": ModelConfig.MODEL_PATH,
        "max_tokens": ModelConfig.MAX_TOKENS,
        "device": device
    }


@app.post("/config")
async def update_config(config: dict):
    """更新配置"""
    try:
        ModelConfig.update_from_dict(config)
        return {"status": "success", "message": "配置已更新"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动SmolVLM Model Service...")
    uvicorn.run(
        app,
        host=ModelConfig.HOST,
        port=ModelConfig.PORT,
        log_level="info"
    )
