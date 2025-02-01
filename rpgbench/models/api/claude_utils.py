import anthropic
import random
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable


class ClaudeCompletion(object):

    def __init__(self, api_keys: None | str | List[str], max_retry: int = 32, retry_sleep: int = 10, error_output: Optional[str] = None):
        if api_keys is not None:
            self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.error_output = error_output

    def _default_verification(self, output):
        return output.startswith("I'm sorry, I can't")

    def completion(self, messages, model, temperature=0.0, max_tokens=4096, top_p=1.0, verification:Optional[Callable]=None):
        if verification is None:
            verification = self._default_verification
        client = anthropic.Anthropic(api_key=self.api_keys[random.randint(0, len(self.api_keys) - 1)])

        if messages[0]['role'] == 'system':
            system = messages[0]['content']
            rest_messages = messages[1:]
        else:
            system = ""
            rest_messages = messages
        output = self.error_output
        for _ in range(self.max_retry):
            try:
                completion = client.messages.create(
                    model=model,
                    system=system,
                    messages=rest_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.content[0].text
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
            except anthropic.RateLimitError as e:
                print(type(e), e)
                time.sleep(self.retry_sleep)
            except anthropic.BadRequestError as e:
                print(messages)
                print(type(e), e)
            except anthropic.InternalServerError as e:
                print(type(e), e)
                time.sleep(self.retry_sleep)
            except KeyError:
                print(type(e), e)
                break
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