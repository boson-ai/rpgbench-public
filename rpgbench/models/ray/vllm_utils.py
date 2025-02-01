import json
import math
import os
import ray
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from queue import Queue
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional
from vllm import LLM, SamplingParams

@ray.remote
class vLLMActor:
    default_max_retry = 3
    default_vllm_kwargs = {
        "model": "/fsx/models/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 32768,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "swap_space": 8,
        "trust_remote_code": True
    }
    default_sampling_params = {
        "n": 1,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 8192,
        "stop": ['<|end_of_text|>', '<|eot_id|>', '<|im_end|>']
    }
    magical_number = 8

    def __init__(self, **kwargs):
        if "max_retry" in kwargs:
            self.max_retry = kwargs.pop("max_retry")
        else:
            self.max_retry = self.default_max_retry
        vllm_kwargs = deepcopy(self.default_vllm_kwargs)
        vllm_kwargs.update(kwargs)
        self.llm = LLM(**vllm_kwargs)
        self._max_model_len = vllm_kwargs["max_model_len"]
    
    def prepare_messages(self, messages: str | List[str] | List[Dict[str, str]] | List[List[Dict[str, str]]]) -> List[str]:
        if isinstance(messages, str):
            text_messages = [messages]
        elif not isinstance(messages, list):
            raise ValueError(f"Invalid type of messages: {type(messages)}")
        elif isinstance(messages[0], str):
            text_messages = messages
        elif isinstance(messages[0], dict):
            text_messages = [self.llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        elif isinstance(messages[0], list):
            text_messages = [self.llm.get_tokenizer().apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        else:
            raise ValueError(f"Invalid type of messages: {type(messages[0])}")
        return text_messages
    
    def is_chat(self, messages: str | List[str] | List[Dict[str, str]] | List[List[Dict[str, str]]]) -> bool:
        if isinstance(messages, str):
            return False
        if isinstance(messages[0], str):
            return False
        return True
    
    def regularize_messages(self, messages: str | List[str] | List[Dict[str, str]] | List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
        if isinstance(messages, str):
            return [messages]
        if isinstance(messages, list) and isinstance(messages[0], dict):
            return [messages]
        return messages
    
    def reorder(self, strings: List[str]):
        indexed_strings = list(enumerate(strings))
        indexed_strings.sort(key=lambda x: x[1])
        sorted_indices, sorted_strings = zip(*indexed_strings)
        # chunk into a total of magical_number chunks
        group_size = (len(sorted_indices) + self.magical_number - 1) // self.magical_number
        group_ends = list(range(group_size, len(sorted_indices), group_size)) + [len(sorted_indices)]
        group_pointers = list(range(0, len(sorted_indices), group_size))
        reordered_indices = []
        reordered_strings = []
        while any([pointer < end for pointer, end in zip(group_pointers, group_ends)]):
            for group_idx, (pointer, end) in enumerate(zip(group_pointers, group_ends)):
                if pointer < end:
                    reordered_indices.append(sorted_indices[pointer])
                    reordered_strings.append(sorted_strings[pointer])
                    group_pointers[group_idx] += 1
        return reordered_indices, reordered_strings

    def restore(self, strings: List[str], indices: List[int]):
        restored_strings = [None] * len(strings)
        for idx, string in zip(indices, strings):
            restored_strings[idx] = string
        return restored_strings
    
    def generate(self, messages: str | List[str]| List[Dict[str, str]] | List[List[Dict[str, str]]], sampling_kwargs: Dict = None, verification_func: Optional[Callable] = None) -> List[List[Dict[str, str]]]:
        _sampling_params = deepcopy(self.default_sampling_params)
        if sampling_kwargs is None:
            sampling_kwargs = {}
        _sampling_params.update(sampling_kwargs)
        sampling_params = SamplingParams(**_sampling_params)
        is_chat = self.is_chat(messages)
        
        messages = self.regularize_messages(messages)
        text_messages = self.prepare_messages(messages)
        if len(text_messages) == 0:
            return []
        original_indices, text_messages = self.reorder(text_messages)
        reordered_messages = [messages[idx] for idx in original_indices]
        results = [[] for _ in range(len(text_messages))]
        initial_outputs = self.llm.generate(text_messages, sampling_params, use_tqdm=False)
        if verification_func is None:
            for message_idx, output in enumerate(initial_outputs):
                for sample_output in output.outputs:
                    results[message_idx].append([{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}])
            return self.restore(results, original_indices)
        remaining_messages = []
        for message_idx, output in enumerate(initial_outputs):
            for sample_output in output.outputs:
                verification_results = verification_func(sample_output.text)
                if isinstance(verification_results, tuple):
                    if not is_chat:
                        warnings.warn("Verification function returns appending messages., but the input is not a chat message. Ignoring the returned appending messages.")
                    retry, appending_messages = verification_results
                    if retry:
                        if is_chat:
                            remaining_messages.append({
                                "messages": reordered_messages[message_idx] + appending_messages,
                                "message_idx": message_idx,
                                "original_length": len(reordered_messages[message_idx])
                            })
                        else:
                            remaining_messages.append({
                                "messages": reordered_messages[message_idx],
                                "message_idx": message_idx,
                                "original_length": len(reordered_messages[message_idx])
                            })
                    else:
                        results[message_idx].append([{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}])
                else:
                    retry = verification_results
                    if retry:
                        remaining_messages.append({
                            "messages": reordered_messages[message_idx],
                            "message_idx": message_idx,
                            "original_length": len(reordered_messages[message_idx])
                        })
                    else:
                        results[message_idx].append([{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}])
        
        if len(remaining_messages) == 0:
            return self.restore(results, original_indices)
        
        retry_sampling_params = deepcopy(sampling_params)
        retry_sampling_params.n = 1
        for _ in range(self.max_retry):
            outputs = self.llm.generate(self.prepare_messages([m['messages'] for m in remaining_messages]), retry_sampling_params, use_tqdm=False)
            new_remaining_messages = []
            for idx, output in enumerate(outputs):
                message_idx = remaining_messages[idx]['message_idx']
                for sample_output in output.outputs:
                    verification_results = verification_func(sample_output.text)
                    if isinstance(verification_results, tuple):
                        retry, appending_messages = verification_results
                        if retry:
                            if is_chat:
                                new_remaining_messages.append({
                                    "messages": remaining_messages[idx]['messages'] + appending_messages,
                                    "message_idx": message_idx,
                                    "original_length": remaining_messages[idx]['original_length']
                                })
                            else:
                                new_remaining_messages.append({
                                    "messages": remaining_messages[idx]['messages'],
                                    "message_idx": message_idx,
                                    "original_length": remaining_messages[idx]['original_length']
                                })
                        else:
                            if is_chat:
                                results[message_idx].append(
                                    remaining_messages[idx]['messages'][remaining_messages[idx]['original_length']:] + \
                                        [{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}]
                                )
                            else:
                                results[message_idx].append(
                                    [{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}]
                                )
                    else:
                        retry = verification_results
                        if retry:
                            new_remaining_messages.append({
                                "messages": remaining_messages[message_idx],
                                "message_idx": message_idx,
                                "original_length": remaining_messages[idx]['original_length']
                            })
                        else:
                            results[message_idx].append([{"role": "assistant", "content": sample_output.text, "logprobs": getattr(sample_output, "cumulative_logprob", None)}])
            if len(new_remaining_messages) == 0:
                break
            remaining_messages = new_remaining_messages
        return self.restore(results, original_indices)
    
    def file_generate(
        self,
        input_file: str,
        output_file: str,
        sampling_kwargs: Dict = None,
        verification_func: Optional[Callable] = None,
        preprocess_func: Optional[Callable] = None,
        postprocess_func: Optional[Callable] = None
    ) -> List[List[Dict[str, str]]]:
        """
        The input_file has to be either a txt file, a json file or a jsonl file. All files should have the same postfix.
        If the file is a text file, it should have one prompt string per line.
        If the file is a json file, it should contain a list.
        If the file is a jsonl file, it should contain a json dictionary on each of its lines.
        """
        postfix = os.path.splitext(input_file)[1]
        messages = []
        if postfix == ".txt":
            with open(input_file, "r") as f:
                messages.extend(f.readlines())
        elif postfix == ".json":
            with open(input_file, "r") as f:
                messages.extend(json.load(f))
        elif postfix == ".jsonl":
            with open(input_file, "r") as f:
                messages.extend([json.loads(line) for line in f])
        else:
            raise ValueError(f"Unsupported file format: '{postfix}'")
        if preprocess_func is not None:
            input_messages = preprocess_func(messages)
        else:
            input_messages = messages
        results = self.generate(input_messages, sampling_kwargs, verification_func)
        if postprocess_func is not None:
            results = postprocess_func(results, messages)
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        return output_file


class vLLMRayCompletion(object):
    def __init__(self, ray_headnode:str = None, tensor_parallel_size:int = 1, num_gpus: int = None, num_cpus:int = None, pythonpath:str = "", **kwargs):
        """
        ray_headnode: The headnode of the Ray cluster. If None, it will start a local Ray cluster.
        tensor_parallel_size: The number of GPUs for each actor.
        num_gpus: The total number of GPUs to use. If None, it will use all available GPUs.
        num_cpus: The total number of CPUs to use. If None, it will use all available CPUs.
        """
        if ray_headnode is None:
            ray.init(local_mode=True, ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        else:
            ray.init(f"ray://{ray_headnode}:10001", ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        resources = ray.cluster_resources()
        if num_gpus is None:
            num_gpus = int(resources.get("GPU", 0))
        if num_cpus is None:
            num_cpus = int(resources.get("CPU", 0))
        num_actors = num_gpus // tensor_parallel_size
        if num_actors == 0:
            raise ValueError("Insufficient GPUs for the given tensor_parallel_size.")
        num_cpus_per_actor = num_cpus // num_actors
        self.actors:List[vLLMActor] = [
            vLLMActor.options(
                **self.define_options(
                    num_gpus_per_actor=tensor_parallel_size,
                    num_cpus_per_actor=num_cpus_per_actor)
                ).remote(tensor_parallel_size=tensor_parallel_size, **kwargs)
            for _ in range(num_actors)
        ]
        self.available_actor_ids = Queue()
        for i in range(num_actors):
            self.available_actor_ids.put(i)

    @staticmethod
    def define_options(
        num_gpus_per_actor: int = 1,
        num_cpus_per_actor: int = 1,
    ):
        return {
            "num_gpus": num_gpus_per_actor,
            "num_cpus": num_cpus_per_actor,
        }
    
    def _completion(self, messages: str|List[str], sampling_kwargs, verification_func: Optional[Callable]):
        try:
            actor_id = self.available_actor_ids.get()
            outputs = ray.get(self.actors[actor_id].generate.remote(messages, sampling_kwargs, verification_func))
        except Exception as e:
            print(e)
            outputs = None
        finally:
            self.available_actor_ids.put(actor_id)
        return outputs
    
    def batch_completion(self, messages: str|List[str]|List[List[Dict[str,str]]], batch_size: int = None, timeout: float = None, verification_func: Optional[Callable] = None,  sampling_kwargs: Dict = None) -> List[str]:
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        if batch_size is None:
            batch_size = (len(messages) + len(self.actors) - 1) // len(self.actors)
        results = [None] * len(messages)
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            for batch_idx in range(0, len(messages), batch_size):
                batch = messages[batch_idx:batch_idx + batch_size]
                future = executor.submit(self._completion, batch, sampling_kwargs, verification_func)
                future_to_index[future] = batch_idx
            for future in tqdm(as_completed(future_to_index, timeout=timeout), total=len(future_to_index)):
                idx = future_to_index[future]
                try:
                    batch_result = future.result()
                    results[idx:idx + len(batch_result)] = batch_result
                except Exception as e:
                    warnings.warn(f"Batch starting at index {idx} failed with exception: {e}")
        return results
    
    def batch_completion_iter(self, messages: str|List[str]|List[List[Dict[str,str]]], batch_size: int = None, timeout: float = None, verification_func: Optional[Callable] = None,  sampling_kwargs: Dict = None):
        if isinstance(messages, str):
            messages = [messages]
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        if batch_size is None:
            batch_size = (len(messages) + len(self.actors) - 1) // len(self.actors)
        # results = [None] * len(messages)
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            for batch_idx in range(0, len(messages), batch_size):
                batch = messages[batch_idx:batch_idx + batch_size]
                future = executor.submit(self._completion, batch, sampling_kwargs, verification_func)
                future_to_index[future] = batch_idx
            for future in tqdm(as_completed(future_to_index, timeout=timeout), total=len(future_to_index)):
                idx = future_to_index[future]
                try:
                    batch_result = future.result()
                    yield idx, batch_result
                except Exception as e:
                    warnings.warn(f"Batch starting at index {idx} failed with exception: {e}")
    
    def _file_completion(self, file: str, sampling_kwargs, verification_func: Optional[Callable], preprocess_func: Optional[Callable], postprocess_func: Optional[Callable]):
        assert "." in os.path.basename(file), "Cannot determine the file type based on the file extension."
        output_file = file.rsplit(".", 1)[0] + ".out." + file.rsplit(".", 1)[1]
        try:
            actor_id = self.available_actor_ids.get()
            outputs = ray.get(self.actors[actor_id].file_generate.remote(
                input_file=file,
                output_file=output_file, 
                sampling_kwargs=sampling_kwargs,
                verification_func=verification_func,
                preprocess_func=preprocess_func,
                postprocess_func=postprocess_func))
        except Exception as e:
            print(e)
            outputs = None
        finally:
            self.available_actor_ids.put(actor_id)
        return outputs
    
    def batch_file_completion(self, files: str|List[str]|List[List[Dict[str,str]]], timeout: float = None, verification_func: Optional[Callable] = None,  sampling_kwargs: Dict = None, preprocess_func: Optional[Callable] = None, postprocess_func: Optional[Callable] = None) -> List[str]:
        """
        Returns a list of output file paths.
        """
        if isinstance(files, str):
            files = [files]
        files = [os.path.abspath(file) for file in files]
        with ThreadPoolExecutor() as executor:
            future_to_index = {}
            for batch_idx in range(len(files)):
                future = executor.submit(self._file_completion, files[batch_idx], sampling_kwargs, verification_func, preprocess_func, postprocess_func)
                future_to_index[future] = batch_idx
            results = [None] * len(future_to_index)
            for future in tqdm(as_completed(future_to_index, timeout=timeout), total=len(future_to_index)):
                idx = future_to_index[future]
                try:
                    batch_result = future.result()
                    results[idx] = batch_result
                except Exception as e:
                    warnings.warn(f"Batch starting at index {idx} failed with exception: {e}")
        return results
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        ray.shutdown()
    
    def shutdown(self):
        ray.shutdown()