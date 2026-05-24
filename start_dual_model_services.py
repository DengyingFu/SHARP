"""
单进程启动两个 FastAPI 模型服务。

目标：
- 同时启动 smolVLM/model_server.py 与 ambiguity/qwen_model_server.py
- 保持两个独立端口
- 只使用一个 Python 进程，从而避免两份 CUDA 上下文
"""

import asyncio
import logging
import signal

import uvicorn

from qwen3.qwen_model_server import ModelConfig as QwenModelConfig
from qwen3.qwen_model_server import app as qwen_app
from smolVLM.model_server import ModelConfig as SmolModelConfig
from smolVLM.model_server import app as smol_app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dual-model-services")


def _build_server(app, host: str, port: int, name: str) -> uvicorn.Server:
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    logger.info("%s 准备启动: http://%s:%s", name, host, port)
    return server


async def _serve_all():
    smol_server = _build_server(
        smol_app,
        SmolModelConfig.HOST,
        SmolModelConfig.PORT,
        "SmolVLM",
    )
    qwen_server = _build_server(
        qwen_app,
        QwenModelConfig.HOST,
        QwenModelConfig.PORT,
        "Qwen Agent",
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(smol_server.serve(), name="smolvlm-server"),
        asyncio.create_task(qwen_server.serve(), name="qwen-server"),
    ]

    logger.info(
        "两个服务已在同一进程中启动: SmolVLM=%s, Qwen=%s",
        SmolModelConfig.PORT,
        QwenModelConfig.PORT,
    )

    await stop_event.wait()

    logger.info("接收到退出信号，准备关闭服务...")
    smol_server.should_exit = True
    qwen_server.should_exit = True
    await asyncio.gather(*tasks)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Start dual model services with custom model paths.")
    parser.add_argument("--qwen-model-path", type=str, default=None, help="Path to the Qwen model directory")
    parser.add_argument("--smol-model-path", type=str, default=None, help="Path to the SmolVLM model directory")
    
    args = parser.parse_args()
    
    if args.qwen_model_path:
        QwenModelConfig.MODEL_PATH = args.qwen_model_path
        logger.info("Override Qwen Model Path: %s", args.qwen_model_path)
        
    if args.smol_model_path:
        SmolModelConfig.MODEL_PATH = args.smol_model_path
        logger.info("Override SmolVLM Model Path: %s", args.smol_model_path)
        
    asyncio.run(_serve_all())


if __name__ == "__main__":
    main()
