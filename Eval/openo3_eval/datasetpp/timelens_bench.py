import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

SUBDATASETS = [
    ("activitynet-timelens.json", "activitynet"),
    ("charades-timelens.json", "charades"),
    ("qvhighlights-timelens.json", "qvhighlights"),
]


def _infer_source_dataset(path: Path) -> str:
    name = path.name.lower()
    for _, source_name in SUBDATASETS:
        if source_name in name:
            return source_name
    return path.stem.replace("-timelens", "")


def _iou(span1: Tuple[float, float], span2: Tuple[float, float]) -> float:
    s1, e1 = span1
    s2, e2 = span2
    if s1 >= e1 or s2 >= e2:
        return 0.0
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    if inter_start >= inter_end:
        return 0.0
    intersection = inter_end - inter_start
    union = (e1 - s1) + (e2 - s2) - intersection
    return intersection / union if union > 0 else 0.0


@DATASET_REGISTRY.register(name=DataSets.timelens_bench)
class TimeLensBench(BaseDatasetPP):
    """TimeLens-Bench: Temporal Grounding evaluation dataset.

    Requires --dataset_dir pointing to the TimeLens-Bench root directory containing:
      - activitynet-timelens.json, charades-timelens.json, qvhighlights-timelens.json
      - videos/{activitynet,charades,qvhighlights}/*.mp4
    """

    dataset_name = DataSets.timelens_bench
    answer_parser = AnswerParsers.timelens_bench_parser

    dataset_dir_env_key = "TIMELENS_BENCH_DATASET_DIR"
    user_prompt_env_key = "TIMELENS_BENCH_USER_PROMPT"
    input_mode_env_key = "TIMELENS_BENCH_INPUT_MODE"
    sample_fps_env_key = "TIMELENS_BENCH_SAMPLE_FPS"
    resize_env_key = "TIMELENS_BENCH_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_dir = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        dataset_name_value = self.dataset_name.value if hasattr(self.dataset_name, "value") else self.dataset_name

        self.dataset = []

        all_samples: List[Tuple[str, str, List[float], str, float]] = []
        counts: Dict[str, Tuple[int, int]] = {source_name: (0, 0) for _, source_name in SUBDATASETS}
        if dataset_dir.is_file():
            anno_specs = [(dataset_dir, _infer_source_dataset(dataset_dir))]
            benchmark_root = dataset_dir.parent
        else:
            anno_specs = [(dataset_dir / anno_file, source_name) for anno_file, source_name in SUBDATASETS]
            benchmark_root = dataset_dir

        for anno_path, source_name in anno_specs:
            counts.setdefault(source_name, (0, 0))
            video_root = benchmark_root / "videos" / source_name

            if not anno_path.exists():
                logger.warning(f"Annotation file not found, skipping: {anno_path}")
                continue

            with open(anno_path, "r", encoding="utf-8") as f:
                raw_annos = json.load(f)

            for vid, raw_anno in raw_annos.items():
                video_path = video_root / (vid + ".mp4")
                if not video_path.exists():
                    count, skipped = counts.get(source_name, (0, 0))
                    counts[source_name] = (count, skipped + 1)
                    continue

                for span, query in zip(raw_anno["spans"], raw_anno["queries"], strict=False):
                    all_samples.append(
                        (source_name, str(video_path), span, query, float(raw_anno.get("duration", 0) or 0))
                    )

        cache_results = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, video_path, _, _, _ in all_samples})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        for source_name, video_path, span, query, duration in all_samples:
            count, skipped = counts.get(source_name, (0, 0))

            pp_meta = {
                "gt_span": span,
                "gt_dataset": dataset_name_value,
                "gt_source_dataset": source_name,
                "gt_question": query,
                "gt_duration": duration,
            }

            build_user_prompt_kwargs = {"query": query}

            if self.is_video_path_input_mode(input_mode):
                prompt = self.build_user_prompt(prompt_replacements=None, **build_user_prompt_kwargs)
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)
                counts[source_name] = (count + 1, skipped)
            else:
                cached = cache_results.get(video_path)
                if not cached:
                    counts[source_name] = (count, skipped + 1)
                    continue

                frame_paths, frame_ts = cached

                image_token_content = "".join(f"<{ts:.1f} seconds><image>" for ts in frame_ts)
                prompt = self.build_user_prompt(
                    prompt_replacements={"<video>": image_token_content},
                    **build_user_prompt_kwargs,
                )
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "images": frame_paths,
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)
                counts[source_name] = (count + 1, skipped)

        for source_name, (count, skipped) in counts.items():
            logger.info(
                f"[Open-o3 Eval] TimeLens source={source_name}: {count} samples loaded, "
                f"{skipped} videos missing/failed (mode={input_mode})"
            )

        logger.info(f"[Open-o3 Eval] TimeLens total: {len(self.dataset)} samples (mode={input_mode})")
        return self.dataset

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            ious: List[float] = []
            truncated_count = 0

            for data in items:
                pred_window, is_truncated = self.parse_answer(data.get("response", ""))

                if is_truncated:
                    truncated_count += 1

                pp = data.get("pp_meta", {})
                gt_span = pp.get("gt_span") or data.get("gt_span") or (-100.0, -100.0)

                ious.append(_iou(pred_window, tuple(gt_span)))

            n = len(items)
            recall = {}
            for threshold in [0.3, 0.5, 0.7]:
                recall[f"R@{threshold}"] = round(sum(1 for i in ious if i >= threshold) * 100 / n, 2) if n > 0 else 0
            miou = round(sum(ious) * 100 / n, 2) if n > 0 else 0

            return {
                "total": n,
                "truncated": truncated_count,
                "truncated_pct": round(truncated_count * 100 / n, 2) if n > 0 else 0,
                **recall,
                "mIoU": miou,
            }

        # Group by TimeLens source split while keeping the benchmark dataset name as timelens_bench.
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for data in data_list:
            pp = data.get("pp_meta", {})
            source = pp.get("gt_source_dataset") or data.get("gt_source_dataset") or pp.get("gt_dataset") or "unknown"
            dataset_name_value = self.dataset_name.value if hasattr(self.dataset_name, "value") else self.dataset_name
            if source == dataset_name_value or source == self.dataset_name:
                source = "unknown"
            by_source.setdefault(source, []).append(data)

        all_metrics: Dict[str, Any] = {"overall": summarize(data_list)}
        for source_name, items in sorted(by_source.items()):
            all_metrics[f"source={source_name}"] = summarize(items)

        return all_metrics

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pred_window, is_truncated = self.parse_answer(result.get("response", ""))
        pp = result.get("pp_meta", {})
        gt_span = pp.get("gt_span") or result.get("gt_span") or (-100.0, -100.0)
        iou = _iou(pred_window, tuple(gt_span))
        return {
            "sample_eval": {
                "metric_type": "temporal_grounding",
                "pred_window": list(pred_window) if isinstance(pred_window, tuple) else pred_window,
                "gold_window": list(gt_span) if isinstance(gt_span, tuple) else gt_span,
                "iou": round(float(iou), 6),
                "truncated": bool(is_truncated),
                "acc@0.3": float(iou >= 0.3),
                "acc@0.5": float(iou >= 0.5),
                "acc@0.7": float(iou >= 0.7),
            }
        }
