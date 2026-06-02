from vllm import LLM
from vllm.sampling_params import SamplingParams
from datetime import datetime, timedelta


class LLMMgr:


    def __init__(self):
        self._last_vllm = None
        self._last_model_name = None
        self._system_prompt = "You are a conversational agent that always answers straight to the point"


    def do_inference(self,
                 user_prompt: str,
                 model_name: str,
                 temperature: float = 0.15,
                 max_tokens: int = 512,
                 system_prompt: str = None) -> str:

        if not user_prompt:
            print("USER PROMPT IS EMPTY")
            raise ValueError("user_prompt cannot be empty or None")

        if not model_name:
            print("MODEL NAME IS EMPTY")
            raise ValueError("model_name cannot be empty or None")

        try:
            system_prompt = self._system_prompt if system_prompt is None else system_prompt

            print(f"LOADING MODEL: {model_name}")
            if model_name == self._last_model_name:
                vllm = self._last_vllm
                print(f"REUSING CACHED vLLM FOR: {model_name}")
            else:
                print(f"INSTANTIATING NEW vLLM FOR: {model_name}")

                #---TODO: make the tensor_parallel_size 
                vllm = LLM(model=model_name, tensor_parallel_size=1)
                self._last_vllm = vllm
                self._last_model_name = model_name

            sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
            print(f"SAMPLING PARAMS: max_tokens={max_tokens}, temperature={temperature}")

            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ]

            print(f"RUNNING vLLM CHAT")
            outputs = vllm.chat(messages, sampling_params=sampling_params)
            print(f"FINISHED vLLM CHAT")

            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty outputs")

            return outputs[0].outputs[0].text

        except ValueError as v:
            print(f"========== VALUE ERROR ===================={v}")
            raise
        except RuntimeError as r:
            print(f"========== RUNTIME ERROR =================={r}")
            raise
        except Exception as e:
            print(f"========================== INFERENCE ERROR: {e} ========================")
            raise RuntimeError(f"do_inference failed: {e}") from e
            if not prompt:
                print("PROMPT IS EMPTY")
                raise ValueError("Prompt cannot be empty or None")