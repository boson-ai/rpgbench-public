

from typing import Literal, Optional, Union
from rpgbench.models.api.claude_utils import ClaudeCompletion
from rpgbench.models.api.deepseek_utils import DeepSeekCompletion
from rpgbench.models.api.gemini_utils import GeminiCompletion
from rpgbench.models.api.openai_utils import OpenAICompletion
from rpgbench.models.ray.vllm_utils import vLLMRayCompletion

def get_model(
    model_type: Literal['claude', 'deepseek', 'gemini', 'openai', 'ray'],
    model_name_or_path: Optional[str] = None,
    api_keys: Optional[str | list[str]] = None,
    ray_headnode: Optional[str] = None,
    ray_n_gpus_per_model: Optional[int] = None,
    ray_n_gpus_total: Optional[int] = None,
    ) -> Union[ClaudeCompletion, DeepSeekCompletion, GeminiCompletion, OpenAICompletion, vLLMRayCompletion]:

    if model_type == 'claude':
        return ClaudeCompletion(api_keys=api_keys)
    elif model_type == 'deepseek':
        return DeepSeekCompletion(api_keys=api_keys)
    elif model_type == 'gemini':
        return GeminiCompletion(api_keys=api_keys)
    elif model_type == 'openai':
        return OpenAICompletion(api_keys=api_keys)
    elif model_type == 'ray':
        return vLLMRayCompletion(
            ray_headnode=ray_headnode,
            model=model_name_or_path,
            tensor_parallel_size=ray_n_gpus_per_model,
            num_gpus=ray_n_gpus_total,
            max_model_len=32768
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")