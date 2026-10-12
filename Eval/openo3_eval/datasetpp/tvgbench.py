import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openo3_eval.datasetpp.timelens_bench import TimeLensBench
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

DEFAULT_TVGBENCH_VIDEO_ROOT = Path(os.environ.get("TVGBENCH_VIDEO_ROOT", "data/benchmarks/TVGBench/videos"))
_SPAN_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*-\s*([-+]?\d+(?:\.\d+)?)\s*$")


def _parse_gt_span(value: Any) -> List[float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None

    if isinstance(value, dict):
        for keys in (("start", "end"), ("start_time", "end_time")):
            if all(key in value for key in keys):
                try:
                    return [float(value[keys[0]]), float(value[keys[1]])]
                except (TypeError, ValueError):
                    return None

    match = _SPAN_PATTERN.match(str(value or ""))
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2))]


def _resolve_annotation_path(dataset_dir: Path) -> Path:
    if dataset_dir.is_file():
        return dataset_dir

    candidates = [
        dataset_dir / "tvgbench.json",
        dataset_dir / "annotations" / "tvgbench.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"[Open-o3 Eval] TVGBench annotation file not found under: {dataset_dir}")


def _candidate_video_roots(anno_path: Path) -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("TVGBENCH_VIDEO_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))

    roots.extend(
        [
            anno_path.parent / "tvgbench_data",
            anno_path.parent / "videos" / "tvgbench_data",
            anno_path.parent.parent / "videos" / "tvgbench_data",
            DEFAULT_TVGBENCH_VIDEO_ROOT,
        ]
    )
    return roots


def _resolve_video_path(anno_path: Path, raw_path: Any, video_roots: List[Path]) -> str | None:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return None

    raw = Path(path_text)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                anno_path.parent / raw,
                anno_path.parent.parent / raw,
            ]
        )
        if len(anno_path.parents) >= 4:
            candidates.append(anno_path.parents[3] / raw)

    basename = raw.name
    candidates.extend(root / basename for root in video_roots)

    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return normalized
    return None


@DATASET_REGISTRY.register(name=DataSets.tvgbench)
class TVGBench(TimeLensBench):
    """TVGBench temporal grounding benchmark in the TimeR1 data format."""

    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "timelens_bench.txt"

    dataset_name = DataSets.tvgbench
    answer_parser = AnswerParsers.timelens_bench_parser

    dataset_dir_env_key = "TVGBENCH_DATASET_DIR"
    user_prompt_env_key = "TVGBENCH_USER_PROMPT"
    input_mode_env_key = "TVGBENCH_INPUT_MODE"
    sample_fps_env_key = "TVGBENCH_SAMPLE_FPS"
    resize_env_key = "TVGBENCH_RESIZE"

    def build_dataset(self) -> List[Dict[str, Any]]:
        anno_path = _resolve_annotation_path(self.get_dataset_dir())
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        dataset_name_value = self.dataset_name.value if hasattr(self.dataset_name, "value") else self.dataset_name
        video_roots = _candidate_video_roots(anno_path)

        with open(anno_path, "r", encoding="utf-8") as f:
            raw_annos = json.load(f)
        if not isinstance(raw_annos, list):
            raise ValueError(f"[Open-o3 Eval] TVGBench annotation must be a list: {anno_path}")

        self.dataset = []
        all_samples: List[Tuple[int, Dict[str, Any], str, List[float], str]] = []
        skipped_missing_video = 0
        skipped_invalid = 0

        for idx, row in enumerate(raw_annos):
            if not isinstance(row, dict):
                skipped_invalid += 1
                continue

            question = str(row.get("question") or "").strip()
            span = _parse_gt_span(row.get("answer")) or _parse_gt_span(row)
            video_path = _resolve_video_path(anno_path, row.get("path"), video_roots)
            if not question or span is None:
                skipped_invalid += 1
                continue
            if not video_path:
                skipped_missing_video += 1
                continue

            all_samples.append((idx, row, video_path, span, question))

        cache_results = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, _, video_path, _, _ in all_samples})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        loaded_by_source: Dict[str, int] = {}
        skipped_frames = 0
        for idx, row, video_path, span, question in all_samples:
            source_dataset = str(
                row.get("dataset_name")
                or (row.get("bins") or {}).get("dataset")
                or row.get("source")
                or "unknown"
            ).strip()
            loaded_by_source[source_dataset] = loaded_by_source.get(source_dataset, 0) + 1

            pp_meta = {
                "gt_span": span,
                "gt_dataset": dataset_name_value,
                "gt_source_dataset": source_dataset,
                "gt_question": question,
                "gt_duration": float(row.get("duration") or 0),
                "gt_answer": row.get("answer"),
                "gt_index": idx,
                "gt_video": Path(str(row.get("path") or video_path)).name,
                "gt_question_type": row.get("question_type"),
                "gt_qsemtype": row.get("qsemtype"),
                "gt_source": row.get("source"),
                "gt_bins": row.get("bins") or {},
            }

            build_user_prompt_kwargs = {"query": question}

            if self.is_video_path_input_mode(input_mode):
                prompt = self.build_user_prompt(prompt_replacements=None, **build_user_prompt_kwargs)
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
                self.dataset.append(item)
                continue

            cached = cache_results.get(video_path)
            if not cached:
                skipped_frames += 1
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

        logger.info(
            f"[Open-o3 Eval] TVGBench loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_invalid} invalid rows, "
            f"{skipped_frames} frame failures (mode={input_mode})"
        )
        for source_name, count in sorted(loaded_by_source.items()):
            logger.info(f"[Open-o3 Eval] TVGBench source={source_name}: {count} candidate samples")
        return self.dataset
