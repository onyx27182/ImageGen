from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class DolphinMgr:


    def __init__(self):
        model_id = "/home/ubuntu/Dolphin-X1-Trinity-Nano"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate(self, prompt: str, max_new_tokens : int = 512) -> str:
        if not prompt:
            print("PROMPT IS EMPTY")
            raise ValueError("Prompt cannot be empty or None")

        try:
            messages = [
                {"role": "system", "content": "You are Dolphin, a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]

            print(f"TYPE {type(max_new_tokens)} =>  {max_new_tokens}")



            print(f"CREATING INPUT TOKEN IDs")
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._model.device)
            print(f"FINISHED CREATING INPUT TOKEN IDs")

            print(f"CREATING OUTPUTS")
            print(f"CREATING OUTPUTS")
            print(f"Model device map: {self._model.hf_device_map}")
            print(f"Input IDs device: {input_ids.device}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            with torch.inference_mode():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.95
                )
            print(f"FINISHED CREATING OUTPUTS === DECODING!!!!")
            text_back =  self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"END DECODING")
            return text_back
        except ValueError:
            print("==========VALUE ERROR ====================")
            raise
        except torch.cuda.OutOfMemoryError as e:
            print("============ OOM ==========================")
            raise RuntimeError("GPU out of memory during generation") from e
        except Exception as e:
            print(f"==========================  MODEL ERROR {e} ========================")
            raise RuntimeError(f"Model generation failed: {e}") from e