import argparse
import os
import subprocess
from typing import List


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_gpu_list() -> List[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        return [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    try:
        ps = subprocess.Popen(("nvidia-smi", "--list-gpus"), stdout=subprocess.PIPE)
        output = subprocess.check_output(("wc", "-l"), stdin=ps.stdout)
        return [str(idx) for idx in range(int(output))]
    except Exception:
        return []


def _partition_cuda_visible_devices_for_rank() -> None:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_list = _get_gpu_list()
    if local_world_size <= 1 or len(gpu_list) <= 1:
        return
    if len(gpu_list) < local_world_size:
        raise ValueError(
            f"[Open-o3 Eval] Not enough GPUs for OpenO3 torchrun: {len(gpu_list)} GPUs < {local_world_size} processes"
        )
    gpu_per_proc = max(1, len(gpu_list) // local_world_size)
    start = gpu_per_proc * local_rank
    assigned = gpu_list[start : start + gpu_per_proc]
    if not assigned:
        raise ValueError(f"[Open-o3 Eval] No GPU assigned for LOCAL_RANK={local_rank}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
    print(
        f"[Open-o3 Eval] OpenO3 LOCAL_RANK={local_rank}/{local_world_size}, "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="[Open-o3 Eval] Evaluate OpenO3 style multi-turn video models")
    parser.add_argument("--model", required=True, help="OpenO3 model name")
    parser.add_argument("--dataset_pp", required=True, help="Dataset plugin name, e.g. video_mme or video_holmes")
    parser.add_argument("--dataset_dir", default="", help="Dataset root directory")
    parser.add_argument("--result_path", default=None, help="Output jsonl path")
    parser.add_argument(
        "--videozerobench_mode",
        default="full_levels",
        choices=("full_levels", "level_3_only"),
        help="Evaluation mode for videozerobench",
    )
    parser.add_argument("--val_dataset_sample", type=int, default=None)
    parser.add_argument("--val_dataset_shuffle", type=_parse_bool, default=False)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--answer_parser", default="")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frames_limit", type=int, default=768)
    parser.add_argument("--max_turn", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--use_tools", type=_parse_bool, default=True)
    parser.add_argument("--do_sample", type=_parse_bool, default=False)
    parser.add_argument("--reuse", action="store_true")
    ns = parser.parse_args()
    from openo3_eval.arguments.video_o3 import VideoO3EvalArguments

    return VideoO3EvalArguments(**vars(ns))


def main() -> None:
    _partition_cuda_visible_devices_for_rank()
    args = parse_args()
    from openo3_eval.pipelines.video_o3_eval import run_video_o3_eval

    run_video_o3_eval(args)


if __name__ == "__main__":
    main()
