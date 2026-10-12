import datetime
import os
import random
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

import openo3_eval.datasetpp
import openo3_eval.parser  # noqa: F401
from openo3_eval.arguments.video_o3 import VideoO3EvalArguments
from openo3_eval.datasetpp import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY
from openo3_eval.runtime import get_logger
from openo3_eval.util import append_jsonl, load_jsonl, merge_rank_results, rank_result_path, strip_video_o3_metadata
from openo3_eval.video_o3.base_model import BaseModel
from openo3_eval.video_o3.config import VIDEO_O3_MODEL_CONFIGS
from openo3_eval.video_o3.message import build_video_o3_message
from openo3_eval.video_o3.model_multi_image import VideoO3MultiImageChat

logger = get_logger()


def get_rank_world_size() -> Tuple[int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def init_distributed_if_needed(world_size: int) -> bool:
    if world_size <= 1:
        return False
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = os.environ.get("OPENO3_EVAL_OPENO3_DIST_BACKEND", "gloo")
        timeout_seconds = int(os.environ.get("OPENO3_EVAL_OPENO3_DIST_TIMEOUT_SECONDS", 12 * 60 * 60))
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout_seconds))
    return True


def barrier(distributed: bool) -> None:
    if not distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def destroy_distributed(distributed: bool) -> None:
    if not distributed:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_result_path(args: VideoO3EvalArguments, rank: int, distributed: bool) -> None:
    if not distributed:
        args.init_result_path()
        return

    import torch.distributed as dist

    result_path = [None]
    if rank == 0:
        args.init_result_path()
        result_path[0] = args.result_path
    dist.broadcast_object_list(result_path, src=0)
    args.result_path = result_path[0]
    assert args.result_path is not None


def _force_dataset_video_path_mode(dataset_pp: str) -> None:
    env_key_by_dataset = {
        "video_mme": "VIDEO_MME_INPUT_MODE",
        "video_mme_v2": "VIDEO_MME_V2_INPUT_MODE",
        "lvbench": "LVBENCH_INPUT_MODE",
        "video_holmes": "VIDEO_HOLMES_INPUT_MODE",
        "charades": "CHARADES_INPUT_MODE",
        "timelens_bench": "TIMELENS_BENCH_INPUT_MODE",
        "tvgbench": "TVGBENCH_INPUT_MODE",
        "videozerobench": "VIDEOZEROBENCH_INPUT_MODE",
        "v_star": "V_STAR_INPUT_MODE",
        "worldsense": "WORLDSENSE_INPUT_MODE",
        "videommmu": "VIDEOMMMU_INPUT_MODE",
        "longvideobench": "LONGVIDEOBENCH_INPUT_MODE",
    }
    env_key = env_key_by_dataset.get(dataset_pp)
    if env_key:
        os.environ[env_key] = "video_path"


def _select_dataset_items(
    args: VideoO3EvalArguments, dataset: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed_items = list(enumerate(dataset))
    if args.val_dataset_shuffle:
        rng = random.Random(args.data_seed)
        rng.shuffle(indexed_items)
    if args.val_dataset_sample is not None:
        indexed_items = indexed_items[: args.val_dataset_sample]
    return indexed_items


def _build_model(args: VideoO3EvalArguments) -> BaseModel:
    if args.model not in VIDEO_O3_MODEL_CONFIGS:
        supported = ", ".join(sorted(VIDEO_O3_MODEL_CONFIGS))
        raise ValueError(f"[Open-o3 Eval] Unsupported OpenO3 model {args.model!r}. Supported: {supported}")

    config = VIDEO_O3_MODEL_CONFIGS[args.model]
    kwargs = config.to_kwargs()
    max_turn = args.max_turn if args.max_turn is not None else config.default_max_turn
    if max_turn is None:
        max_turn = 8
    kwargs.update(
        {
            "fps": args.fps,
            "max_turn": max_turn,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "use_tools": args.use_tools,
            "do_sample": args.do_sample,
        }
    )
    if not os.path.exists(str(kwargs["model_path"])):
        raise FileNotFoundError(f"[Open-o3 Eval] OpenO3 model path not found: {kwargs['model_path']}")
    return VideoO3MultiImageChat(**kwargs)


def _build_eval_tasks(
    args: VideoO3EvalArguments,
    dataset_pp: BaseDatasetPP,
    item: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if hasattr(dataset_pp, "build_video_o3_tasks"):
        return list(dataset_pp.build_video_o3_tasks(args, item))

    message, frame_meta = build_video_o3_message(args, item, args.dataset_pp)
    return [{"task_name": "default", "message": message, "frame_meta": frame_meta, "skip": False}]


def _merge_eval_task_results(
    dataset_pp: BaseDatasetPP,
    item: Dict[str, Any],
    task_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if hasattr(dataset_pp, "merge_video_o3_task_results"):
        return dict(dataset_pp.merge_video_o3_task_results(item, task_results))

    task = task_results[0]
    return {
        "response": task["response"],
        "dialogue_turns": task.get("dialogue_turns") or [],
        "video_o3": {
            "turn_count": task.get("turn_count"),
            "inference_time": task.get("inference_time"),
            "inference_time_generate": task.get("inference_time_generate"),
            "raw_response": task.get("raw_response"),
            "frame_meta": task.get("frame_meta"),
            "dialogue_turns": task.get("dialogue_turns") or [],
        },
    }


def _run_task_with_model_overrides(
    model: BaseModel,
    task: Dict[str, Any],
    *,
    dataset_name: str,
) -> str:
    override_keys = [key for key in ("max_turn", "use_tools", "system_prompt", "tg_system_prompt") if key in task]
    original_values = {key: getattr(model, key) for key in override_keys}
    try:
        for key in override_keys:
            setattr(model, key, task[key])
        return model.generate(message=task["message"], dataset=dataset_name)
    finally:
        for key, value in original_values.items():
            setattr(model, key, value)


def _teardown_model(model: BaseModel | None) -> None:
    if model is None:
        return
    for attr in ("model", "processor"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_video_o3_eval(args: VideoO3EvalArguments) -> List[Dict[str, Any]]:
    rank, world_size = get_rank_world_size()
    distributed = init_distributed_if_needed(world_size)

    try:
        broadcast_result_path(args, rank, distributed)
        assert args.result_path is not None

        _force_dataset_video_path_mode(args.dataset_pp)
        dataset_pp: BaseDatasetPP = DATASET_REGISTRY.get(args.dataset_pp)(args)
        dataset = dataset_pp.build_dataset()
        indexed_items = _select_dataset_items(args, dataset)
        rank_items = indexed_items[rank::world_size]

        rank_path = rank_result_path(args.result_path, rank, world_size)
        if not args.reuse and os.path.exists(rank_path):
            os.remove(rank_path)
        done_indices = (
            {int(row.get("_openo3_eval_sample_idx", -1)) for row in load_jsonl(rank_path)} if args.reuse else set()
        )

        logger.info(
            f"[Open-o3 Eval] OpenO3 rank {rank}/{world_size}: {len(rank_items)} samples "
            f"({len(indexed_items)} total selected). result_path={rank_path}"
        )
        model = _build_model(args)

        for sample_idx, item in tqdm(
            rank_items, desc=f"[Open-o3 Eval] OpenO3 rank {rank}", disable=len(rank_items) == 0
        ):
            if sample_idx in done_indices:
                continue
            task_results: List[Dict[str, Any]] = []
            for task in _build_eval_tasks(args, dataset_pp, item):
                if task.get("skip", False):
                    task_results.append(
                        {
                            "task_name": task.get("task_name", "default"),
                            "response": None,
                            "raw_response": None,
                            "frame_meta": task.get("frame_meta"),
                            "error": task.get("error"),
                            "turn_count": None,
                            "inference_time": None,
                            "inference_time_generate": None,
                        }
                    )
                    continue

                raw_response = _run_task_with_model_overrides(model, task, dataset_name=args.dataset_pp)
                response, response_meta = strip_video_o3_metadata(raw_response)
                task_results.append(
                    {
                        "task_name": task.get("task_name", "default"),
                        "response": response,
                        "raw_response": raw_response,
                        "frame_meta": task.get("frame_meta"),
                        "error": task.get("error"),
                        **response_meta,
                    }
                )

            merged = _merge_eval_task_results(dataset_pp, item, task_results)
            result = {
                "_openo3_eval_sample_idx": sample_idx,
                **merged,
                **item,
            }
            result.update(dataset_pp.build_sample_eval(item, result))
            append_jsonl(rank_path, result)

        _teardown_model(model)
        model = None
        barrier(distributed)

        if rank == 0:
            rows = merge_rank_results(args.result_path, world_size) if world_size > 1 else load_jsonl(args.result_path)
            logger.info(f"[Open-o3 Eval] OpenO3 results saved to: {args.result_path}")
            metrics = dataset_pp.calc_metric(rows)
            dataset_pp.print_metric(metrics, args.dataset_pp)
            return rows

        return []
    finally:
        destroy_distributed(distributed)
