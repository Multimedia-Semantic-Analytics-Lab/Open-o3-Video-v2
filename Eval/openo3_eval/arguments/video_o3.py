import os
from dataclasses import dataclass
from typing import Optional

from openo3_eval.runtime import to_abspath


@dataclass
class VideoO3EvalArguments:
    model: str
    dataset_pp: str
    dataset_dir: str = ""
    result_path: Optional[str] = None
    videozerobench_mode: str = "full_levels"
    val_dataset_sample: Optional[int] = None
    val_dataset_shuffle: bool = False
    data_seed: int = 42
    answer_parser: str = ""
    fps: float = 1.0
    frames_limit: int = 768
    max_turn: Optional[int] = None
    max_new_tokens: int = 4096
    temperature: float = 0.01
    top_p: float = 0.001
    top_k: int = 1
    repetition_penalty: float = 1.0
    use_tools: bool = True
    do_sample: bool = False
    reuse: bool = False

    def init_result_path(self) -> None:
        if self.result_path is not None:
            if not self.result_path.endswith(".jsonl"):
                raise ValueError(f"[Open-o3 Eval] result_path {self.result_path} is not a jsonl file")
            result_dir = os.path.dirname(self.result_path)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            self.result_path = to_abspath(self.result_path)
            return

        safe_model = self.model.replace("/", "_")
        result_dir = to_abspath(os.path.join("openo3_eval_result", safe_model, "openo3_eval"))
        os.makedirs(result_dir, exist_ok=True)
        import datetime as dt

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.result_path = os.path.join(result_dir, f"{timestamp}.jsonl")
