from __future__ import annotations

import logging
import subprocess
from abc import abstractmethod
from typing import Any, ClassVar

import torch
from transformers import StoppingCriteria


class BaseModel:
    INTERLEAVE = False
    VIDEO_LLM = False
    allowed_types: ClassVar[list[str]] = ["text", "image", "video"]
    allowed_meta_types: ClassVar[list[str]] = ["question_type", "sample_n_frames", "raw_video"]

    def __init__(self) -> None:
        self.dump_image_func = None

    def use_custom_prompt(self, dataset: Any) -> bool:
        return False

    @abstractmethod
    def build_prompt(self, line: Any, dataset: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def set_dump_image(self, dump_image_func: Any) -> None:
        self.dump_image_func = dump_image_func

    def dump_image(self, line: Any, dataset: Any) -> Any:
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message: list[dict[str, Any]], dataset: Any = None) -> str:
        raise NotImplementedError

    def check_content(self, msgs: Any) -> str:
        if isinstance(msgs, str):
            return "str"
        if isinstance(msgs, dict):
            return "dict"
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == "str" for t in types):
                return "liststr"
            if all(t == "dict" for t in types):
                return "listdict"
        return "unknown"

    def preproc_content(self, inputs: Any) -> list[dict[str, Any]] | None:
        if self.check_content(inputs) == "str":
            return [{"type": "text", "value": inputs}]
        if self.check_content(inputs) == "dict":
            assert "type" in inputs and "value" in inputs
            return [inputs]
        if self.check_content(inputs) == "liststr":
            return [{"type": "text", "value": s} for s in inputs]
        if self.check_content(inputs) == "listdict":
            for item in inputs:
                assert "type" in item and "value" in item
            return inputs
        return None

    def generate(self, message: Any, dataset: Any = None) -> str:
        assert self.check_content(message) in ["str", "dict", "liststr", "listdict"], f"Invalid input type: {message}"
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == "listdict"
        for item in message:
            assert item["type"] in (self.allowed_types + self.allowed_meta_types), f"Invalid input type: {item['type']}"
        return self.generate_inner(message, dataset)

    def chat(self, messages: list[dict[str, Any]], dataset: Any = None) -> str:
        assert hasattr(self, "chat_inner"), "The API model should has the `chat_inner` method."
        for msg in messages:
            assert isinstance(msg, dict) and "role" in msg and "content" in msg, msg
            assert self.check_content(msg["content"]) in ["str", "dict", "liststr", "listdict"], msg
            msg["content"] = self.preproc_content(msg["content"])

        while len(messages):
            try:
                return self.chat_inner(messages, dataset=dataset)
            except Exception as e:
                logging.info(f"{type(e)}: {e}")
                messages = messages[1:]
                while len(messages) and messages[0]["role"] != "user":
                    messages = messages[1:]
        return "Chat Mode: Failed with all possible conversation turns."


def listinstr(items: list[str], text: str) -> bool:
    return any(item in text for item in items)


def get_gpu_memory() -> list[int]:
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        output = subprocess.check_output(command.split()).decode("ascii").splitlines()[1:]
        return [int(line.split()[0]) for line in output if line.strip()]
    except Exception as e:
        print(f"{type(e)}: {e}")
        return []


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1"
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        return any(keyword in outputs for keyword in self.keywords)
