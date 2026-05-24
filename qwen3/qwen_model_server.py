"""
Qwen Agent Model Service
提供 FastAPI 服务来单独加载和推理 language_module_image_only.py 里的 Qwen 模型。
"""

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from qwen_agent.agents import Assistant


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in os.sys.path:
    os.sys.path.insert(0, str(project_root))

from ambiguity.modules.VLM_tools import MyImageGenFullSFT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

bot = None
device = "cuda" if torch.cuda.is_available() else "cpu"
_request_count = 0


class ModelConfig:
    """模型配置"""

    MODEL_PATH = "weights/full"
    HOST = "0.0.0.0"
    PORT = 1235
    MAX_NEW_TOKENS = 512
    TOP_P = 0.8
    TOP_K = 20
    TEMPERATURE = 0.1
    REPETITION_PENALTY = 1.0
    SYSTEM_PROMPT = "You are the reasoning module of a robotic arm. You receive: 1) a user instruction; 2) an unordered list of available object names. Your task is to decide which objects from the list are needed."
    TORCH_DTYPE = "bfloat16"
    EMPTY_CACHE_EVERY_N_REQUESTS = 0


class InferenceRequest(BaseModel):
    """推理请求"""

    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None
    use_tools: Optional[bool] = False
    image_path: Optional[str] = None


class InferenceResponse(BaseModel):
    """推理响应"""

    status: str
    result: str
    error: Optional[str] = None
    thinking: Optional[str] = None
    trace: Optional[List[Dict[str, str]]] = None


def _normalize_messages(
    messages: List[Dict[str, str]],
    request_system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """规范化消息：最多一个 system，且 system（若存在）必须在首位。"""

    normalized_messages: List[Dict[str, str]] = []
    existing_system_content: Optional[str] = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if existing_system_content is None and content:
                existing_system_content = content
            continue
        normalized_messages.append(msg)

    system_content = existing_system_content or request_system_prompt
    if system_content:
        return [{"role": "system", "content": system_content}] + normalized_messages
    return normalized_messages


def _resolve_torch_dtype():
    dtype_name = (ModelConfig.TORCH_DTYPE or "").lower().strip()
    if dtype_name in {"bf16", "bfloat16"}:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device == "cuda" else torch.float32
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    return None


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


def initialize_model() -> bool:
    """初始化 Qwen Agent 模型"""

    global bot
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dtype = _resolve_torch_dtype()
        llm_cfg = {
            "model_type": "transformers",
            "model": ModelConfig.MODEL_PATH,
            "api_key": "EMPTY",
            "device": device,
            "generate_cfg": {
                "top_p": ModelConfig.TOP_P,
                "top_k": ModelConfig.TOP_K,
                "temperature": ModelConfig.TEMPERATURE,
                "repetition_penalty": ModelConfig.REPETITION_PENALTY,
                "max_new_tokens": ModelConfig.MAX_NEW_TOKENS,
            },
        }
        if dtype is not None:
            llm_cfg["torch_dtype"] = dtype

        logger.info("从 %s 加载 Qwen Agent 模型...", ModelConfig.MODEL_PATH)
        bot = Assistant(
            llm=llm_cfg,
            system_message=ModelConfig.SYSTEM_PROMPT,
            function_list=["smolvlm"],
        )
        logger.info("Qwen Agent 模型初始化成功")
        return True
    except Exception as exc:
        logger.error("Qwen Agent 模型初始化失败: %s", exc, exc_info=True)
        return False


def _extract_thinking_and_content(text: str):
    """提取 thinking 与最终文本"""

    thinking = ""
    if not text:
        return "", ""

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    clean_content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, clean_content


def _run_inference(messages: List[Dict[str, str]], use_tools: bool = False, image_path: Optional[str] = None):
    """执行推理；可选启用工具调用。"""

    if bot is None:
        raise RuntimeError("模型未初始化")

    if image_path:
        MyImageGenFullSFT.current_img_path = image_path

    with torch.inference_mode():
        if use_tools:
            response_list = []
            for res in bot.run(messages=messages, response_mode="safe"):
                response_list = res

            final_content = ""
            trace = []
            if response_list:
                final_content = response_list[-1].get("content", "")
                for msg in response_list:
                    role = msg.get("role")
                    if role == "user":
                        continue
                    trace.append({"role": role or "assistant", "content": msg.get("content", "")})

            thinking, content = _extract_thinking_and_content(final_content)
            return content, thinking, trace

        if hasattr(bot, "llm"):
            response_generator = bot.llm.chat(messages=messages, functions=None)
            final_content = ""
            for response in response_generator:
                if isinstance(response, list) and response:
                    final_content = response[-1].get("content", "")
                elif isinstance(response, dict):
                    final_content = response.get("content", "")
                else:
                    final_content = str(response)
            thinking, content = _extract_thinking_and_content(final_content)
            return content, thinking, []

        response_list = []
        for res in bot.run(messages=messages, response_mode="safe"):
            response_list = res

        final_content = ""
        if response_list:
            final_content = response_list[-1].get("content", "")

        thinking, content = _extract_thinking_and_content(final_content)
        return content, thinking, []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动应用，初始化 Qwen Agent 模型...")
    if not initialize_model():
        raise RuntimeError("Qwen Agent 模型初始化失败")

    yield

    logger.info("应用关闭，清理资源...")


app = FastAPI(
    title="Qwen Agent Model Service",
    description="提供 Qwen Agent 本地模型推理服务",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    allocated_gb = 0.0
    reserved_gb = 0.0
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3

    return {
        "status": "ok",
        "device": device,
        "model_loaded": bot is not None,
        "model_path": ModelConfig.MODEL_PATH,
        "cuda_allocated_gb": round(allocated_gb, 3),
        "cuda_reserved_gb": round(reserved_gb, 3),
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    try:
        if bot is None:
            raise HTTPException(status_code=503, detail="模型未初始化")

        if request.messages:
            messages = request.messages
        elif request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        else:
            raise HTTPException(status_code=400, detail="prompt 或 messages 至少需要一个")

        messages = _normalize_messages(messages, request.system_prompt)

        content, thinking, trace = await run_in_threadpool(
            _run_inference,
            messages,
            bool(request.use_tools),
            request.image_path,
        )
        return InferenceResponse(status="success", result=content, thinking=thinking, trace=trace)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("推理失败: %s", exc, exc_info=True)
        return InferenceResponse(status="error", result="", error=str(exc))
    finally:
        _maybe_release_cuda_cache()


if __name__ == "__main__":
    import uvicorn

    logger.info("启动 Qwen Agent Model Service...")
    uvicorn.run(
        app,
        host=ModelConfig.HOST,
        port=ModelConfig.PORT,
        log_level="info",
    )
