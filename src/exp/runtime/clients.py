import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from exp.runtime.loop import canonical_json
from exp.util.cache import DiskCache


class FakeClient:
    """A scripted or callable fake client for unit testing."""
    def __init__(self, responses: Optional[List[str]] = None, response_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None):
        self.responses = list(responses) if responses else []
        self.response_fn = response_fn
        self.call_history: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]], schema: Any) -> Tuple[str, Dict[str, Any]]:
        self.call_history.append(messages)
        if self.response_fn:
            text = self.response_fn(messages)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            raise RuntimeError("FakeClient ran out of responses")
        usage = {
            "input_tokens": 10,
            "output_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return text, usage


class GeminiClient:
    """Client for Google Gemini API."""
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache: Optional[DiskCache] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.use_cache = use_cache
        self.cache = cache or DiskCache()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete(self, messages: List[Dict[str, str]], schema: Any) -> Tuple[str, Dict[str, Any]]:
        if self.use_cache:
            req_hash = DiskCache.compute_hash(messages, schema, self.model)
            cached = self.cache.get(req_hash)
            if cached:
                return cached

        from google.genai import types

        # Format messages into system instructions + turns
        system_instruction = None
        contents: List[types.Content] = []

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

        # Google GenAI API protocol requires the final turn to be from the user
        if contents and contents[-1].role == "model":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text="TOOL_RESULT: continue")]))

        client = self._get_client()
        from exp.schema import UnifiedStep
        from tenacity import retry, stop_after_attempt, wait_exponential

        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=UnifiedStep,
            temperature=0.0,
            system_instruction=system_instruction,
        )

        @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
        def _call_gemini():
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=cfg,
            )

        resp = _call_gemini()

        text = resp.text or "{}"
        u_meta = resp.usage_metadata
        in_tokens = getattr(u_meta, "prompt_token_count", 0) or 0
        out_tokens = getattr(u_meta, "candidates_token_count", 0) or 0
        cached_tokens = getattr(u_meta, "cached_content_token_count", 0) or 0

        usage = {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached_tokens,
        }

        if self.use_cache:
            self.cache.set(req_hash, text, usage)

        return text, usage


class VllmClient:
    """Client for local vLLM OpenAI-compatible server with structured decoding."""
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "base",
        api_key: str = "EMPTY",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def complete(self, messages: List[Dict[str, str]], schema: Any) -> Tuple[str, Dict[str, Any]]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "step", "schema": schema}
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        choice = response.choices[0]
        text = choice.message.content or "{}"
        u = response.usage
        usage = {
            "input_tokens": u.prompt_tokens if u else 0,
            "output_tokens": u.completion_tokens if u else 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return text, usage


class LocalHfClient:
    """Client for local PyTorch + PEFT inference directly on GPU."""
    _instance_cache = {}

    def __init__(self, adapter_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.adapter_path = adapter_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is None:
            cache_key = (self.base_model, str(self.adapter_path))
            if cache_key in LocalHfClient._instance_cache:
                self.model, self.tokenizer = LocalHfClient._instance_cache[cache_key]
            else:
                from pathlib import Path
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import PeftModel

                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    quantization_config=bnb,
                    device_map="auto",
                    dtype=torch.bfloat16,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                adapter_p = Path(self.adapter_path)
                if (adapter_p / "adapter_model.safetensors").exists():
                    self.model = PeftModel.from_pretrained(base, str(adapter_p))
                else:
                    self.model = base
                LocalHfClient._instance_cache[cache_key] = (self.model, self.tokenizer)

    def complete(self, messages: List[Dict[str, str]], schema: Any = None) -> Tuple[str, Dict[str, Any]]:
        self._load()
        from pathlib import Path
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        in_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        gen_tokens = out[0][in_tokens:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        out_tokens = len(gen_tokens)

        usage = {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return text, usage
