import json5
import os
import time

from smolVLM.client import SmolVLMClient
from qwen_agent.tools.base import BaseTool, register_tool


smol_client_instance = SmolVLMClient()


@register_tool('smolvlm')
class MyImageGenFullSFT(BaseTool):
    description = 'Identify object names from feature/position cues'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'An English question to resolve an object by feature or position, e.g., What is the object on the left? or Which is the red object?',
        'required': True
    }]
    current_img_path = None

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json5.loads(params)
            if isinstance(params_dict, str):
                params_dict = json5.loads(params_dict)
        except Exception:
            params_dict = params if isinstance(params, dict) else {}

        prompt = params_dict.get('prompt', '')
        image_path = MyImageGenFullSFT.current_img_path
        print(f"Tool Call: {prompt}")

        if not image_path or not os.path.exists(image_path):
            return json5.dumps(
                {'response': f"Error: image_path is invalid: {image_path}"},
                ensure_ascii=False,
            )

        health = smol_client_instance.health_check()
        if health.get("status") != "ok":
            return json5.dumps(
                {'response': f"Error: SmolVLM service unavailable: {health.get('message', health)}"},
                ensure_ascii=False,
            )

        start = time.time()
        response = smol_client_instance.get_smolvlm_response(prompt=prompt, image_path=image_path)
        elapsed = time.time() - start
        print(f"Tool Return ({elapsed:.2f}s): {response}")
        return json5.dumps({'response': response}, ensure_ascii=False)
