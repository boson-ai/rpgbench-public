import google.generativeai as genai
import random
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Callable


class GeminiCompletion(object):

    def __init__(self, api_keys: Optional[Union[str, List[str]]], max_retry: int = 32, retry_sleep: int = 10, error_output: Optional[str] = None):
        if api_keys is not None:
            self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.error_output = error_output
    
    def _default_verification(self, output):
        return output.startswith("I'm sorry, I can't")
    
    def _convert_assistant_to_model(self, messages):
        system_message = ""
        for message in messages:
            if message['role'] == 'system':
                system_message += f"SYSTEM MESSAGE: {message['content']}\n"
            if message['role'] == 'assistant':
                message['role'] = 'model'
        
        if system_message:
            messages[0]['content'] = system_message + messages[0]['content']
        return messages
    
    def _combine_system_with_user(self, messages):
        combined_messages = []
        system_content = ""
        
        for message in messages:
            if message['role'] == 'system':
                system_content += message['content'] + " "
            else:
                if system_content and message['role'] == 'user':
                    message['content'] = system_content + message['content']
                    system_content = ""
                combined_messages.append(message)
        
        return combined_messages

    def _format_messages_for_gemini(self, messages):
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": message["role"],
                "parts": [{"text": message["content"]}]
            })
        return formatted_messages

    def completion(self, messages, model, temperature=0.0, max_tokens=4096, top_p=1.0, verification: Optional[Callable] = None):
        if verification is None:
            verification = self._default_verification

        messages = self._convert_assistant_to_model(messages)
        messages = self._combine_system_with_user(messages)
        messages = self._format_messages_for_gemini(messages)
        
        genai.configure(api_key=self.api_keys[random.randint(0, len(self.api_keys) - 1)])
        model_instance = genai.GenerativeModel(model)
        output = self.error_output
        
        for _ in range(self.max_retry):
            try:
                response = model_instance.generate_content(
                    messages,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "top_p": top_p
                    }
                )
                output = response.text
                verification_result = verification(output)
                
                if isinstance(verification_result, tuple):
                    retry, appending_messages = verification_result
                    if retry:
                        messages += appending_messages
                        continue
                    else:
                        break
                else:
                    retry = verification_result
                    if retry:
                        continue
                    else:
                        break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.retry_sleep)
        return output

    def batch_completion(self, batch, model:str, num_concurrent_jobs:int=10, temperature:float=0.0, max_tokens:int=4096, top_p=1.0, verification:Optional[Callable]=None):
        with ThreadPoolExecutor(num_concurrent_jobs) as executor:
            futures = {executor.submit(self.completion, example, model, temperature, max_tokens, top_p, verification): i for i, example in enumerate(batch)}
            results = [None for _ in range(len(futures))]
            for future in tqdm.tqdm(as_completed(futures), total=len(batch)):
                example_idx = futures[future]
                results[example_idx] = future.result()
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass