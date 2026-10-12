import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()


def _load_and_format_subtitle(subtitle_path: str | Path, max_chars: int = 12000) -> str:
    if not subtitle_path or not os.path.exists(subtitle_path):
        return ""

    subtitle_path = str(subtitle_path)
    if subtitle_path.endswith(".jsonl"):
        texts: List[str] = []
        with open(subtitle_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(data.get("text") or "").strip()
                if not text:
                    continue
                if texts and texts[-1] == text:
                    continue
                texts.append(text)
        return "\n".join(texts)[:max_chars]

    try:
        import pysrt

        subs = pysrt.open(subtitle_path, encoding="utf-8")
    except Exception:
        try:
            import pysrt

            subs = pysrt.open(subtitle_path)
        except Exception as e:
            logger.warning(f"[Open-o3 Eval] Failed to parse subtitle file {subtitle_path}: {e}")
            return ""

    text_blocks: List[str] = []
    for sub in subs:
        clean_text = re.sub(r"<[^>]+>", "", str(sub.text or "")).replace("\\N", " ").replace("\n", " ").strip()
        if not clean_text:
            continue
        if text_blocks and text_blocks[-1] == clean_text:
            continue
        text_blocks.append(clean_text)
    return "\n".join(text_blocks)[:max_chars]


def _parse_option_lines(raw_options: Any) -> List[str]:
    if isinstance(raw_options, list):
        return [str(option).strip() for option in raw_options if str(option).strip()]

    option_text = str(raw_options or "").strip()
    if not option_text:
        return []

    lines = [line.strip() for line in option_text.splitlines() if line.strip()]
    if lines:
        return lines

    matches = re.findall(r"([A-H]\.\s*.*?)(?=(?:\s+[A-H]\.\s)|$)", option_text, flags=re.DOTALL)
    return [match.strip() for match in matches]


def _get_logic_score(scores: List[int], group_structure: str) -> float:
    group_structure_text = str(group_structure or "").replace(" ", "")
    last_correct_idx = -1
    for idx, value in enumerate(scores):
        if value:
            last_correct_idx = idx
        else:
            break

    if group_structure_text == "[1,2,3,4]":
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_text == "[1,[2,3],4]":
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and len(scores) > 2 and scores[2]:
            last_correct_idx += 1
    elif group_structure_text == "[[1,2],3,4]":
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and len(scores) > 1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"[Open-o3 Eval] Unknown VideoMME-v2 group_structure: {group_structure!r}")

    return score_map.get(last_correct_idx + 1, 0.0)


def _get_relevance_scores(scores: List[int]) -> Tuple[float, float]:
    correct_count = sum(scores)
    exp_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    return exp_map.get(correct_count, 0.0), correct_count * 25.0


@DATASET_REGISTRY.register(name=DataSets.video_mme_v2)
class VideoMMEv2(BaseDatasetPP):
    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "video_mme_v2.txt"
    dataset_name = DataSets.video_mme_v2
    answer_parser = AnswerParsers.video_mme_parser

    dataset_dir_env_key = "VIDEO_MME_V2_DATASET_DIR"
    user_prompt_env_key = "VIDEO_MME_V2_USER_PROMPT"
    input_mode_env_key = "VIDEO_MME_V2_INPUT_MODE"
    sample_fps_env_key = "VIDEO_MME_V2_SAMPLE_FPS"
    resize_env_key = "VIDEO_MME_V2_RESIZE"

    def _load_rows(self, dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
        if dataset_path.is_file():
            benchmark_root = dataset_path.parent
            if dataset_path.suffix == ".json":
                with open(dataset_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                if not isinstance(rows, list):
                    raise ValueError(f"[Open-o3 Eval] VideoMME-v2 json must contain a list of rows: {dataset_path}")
                return rows, dataset_path, benchmark_root
            if dataset_path.suffix == ".parquet":
                df = pd.read_parquet(dataset_path)
                return df.to_dict(orient="records"), dataset_path, benchmark_root
            raise ValueError(f"[Open-o3 Eval] Unsupported VideoMME-v2 annotation file: {dataset_path}")

        candidates = [
            dataset_path / "Video-MME-v2_fullset.json",
            dataset_path / "test.parquet",
            dataset_path / "raw" / "test.parquet",
        ]
        anno_path = next((path for path in candidates if path.exists()), None)
        if anno_path is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"[Open-o3 Eval] VideoMME-v2 data file not found. searched: {searched}")
        if anno_path.suffix == ".json":
            with open(anno_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            return rows, anno_path, dataset_path

        df = pd.read_parquet(anno_path)
        return df.to_dict(orient="records"), anno_path, dataset_path

    def _resolve_media_paths(self, row: Dict[str, Any], benchmark_root: Path) -> Tuple[str, str, str]:
        video_id = str(row.get("video") or row.get("video_id") or row.get("videoID") or "").strip()
        if video_id.isdigit():
            video_id = video_id.zfill(3)

        video_path = str(row.get("video_path") or "").strip()
        if video_path:
            if not os.path.isabs(video_path):
                video_path = str(benchmark_root / video_path)
        else:
            video_path = str(benchmark_root / "videos" / f"{video_id}.mp4")

        subtitle_path = str(row.get("subtitle_path") or "").strip()
        if subtitle_path:
            if not os.path.isabs(subtitle_path):
                subtitle_path = str(benchmark_root / subtitle_path)
        else:
            subtitle_path = str(benchmark_root / "subtitles" / "subtitle" / f"{video_id}.jsonl")

        return video_id, video_path, subtitle_path

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        asr_max_chars = int(os.environ.get("VIDEO_MME_V2_ASR_MAX_CHARS", "12000"))
        prompt_template = self.get_user_prompt_template()
        need_asr = "{asr}" in prompt_template

        self.dataset = []
        rows, anno_path, benchmark_root = self._load_rows(dataset_path)

        all_items: List[Tuple[Dict[str, Any], str, List[str], str, str, str]] = []
        skipped_missing_video = 0
        for row in rows:
            video_id, video_path, subtitle_path = self._resolve_media_paths(row, benchmark_root)
            if not video_path or not os.path.exists(video_path):
                skipped_missing_video += 1
                continue

            candidates = _parse_option_lines(row.get("candidates") or row.get("options"))
            if not candidates:
                logger.warning(
                    f"[Open-o3 Eval] VideoMME-v2 row has no valid options, skip question_id={row.get('question_id')!r}"
                )
                continue

            asr_info = ""
            if need_asr:
                asr_info = _load_and_format_subtitle(subtitle_path, max_chars=asr_max_chars)
            if not asr_info:
                asr_info = "No ASR info available."

            all_items.append(
                (
                    row,
                    video_path,
                    candidates,
                    "\n".join(candidates),
                    str(row.get("answer") or "").strip().upper(),
                    asr_info,
                )
            )

        cache_results: Dict[str, Tuple[List[str], List[float]]] = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, video_path, _, _, _, _ in all_items})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        skipped_frames = 0
        for row, video_path, candidates, choice, answer, asr_info in all_items:
            video_id = str(row.get("video") or row.get("video_id") or row.get("videoID") or "").strip()
            if video_id.isdigit():
                video_id = video_id.zfill(3)

            pp_meta = {
                "gt_answer": answer,
                "gt_candidates": candidates,
                "gt_question": str(row.get("question") or "").strip(),
                "gt_options_text": choice,
                "gt_question_id": str(row.get("question_id") or "").strip(),
                "gt_video": video_id,
                "gt_group_type": str(row.get("group_type") or "").strip(),
                "gt_group_structure": str(row.get("group_structure") or "").strip(),
                "gt_level": "" if row.get("level") is None else str(row.get("level")),
                "gt_second_head": "" if row.get("second_head") is None else str(row.get("second_head")),
                "gt_third_head": "" if row.get("third_head") is None else str(row.get("third_head")),
                "gt_url": str(row.get("url") or "").strip(),
            }

            prompt_kwargs = {
                "question": str(row.get("question") or "").strip(),
                "choice": choice,
                "asr": asr_info,
            }

            if self.is_video_path_input_mode(input_mode):
                prompt = self.build_user_prompt(**prompt_kwargs)
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
                **prompt_kwargs,
            )
            item = {
                "messages": [{"role": "user", "content": prompt}],
                "images": frame_paths,
                "pp_meta": pp_meta,
            }
            self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] VideoMME-v2 loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_frames} frame failures "
            f"(mode={input_mode})"
        )
        return self.dataset

    @staticmethod
    def _group_question_order(question_id: str) -> Tuple[int, str]:
        match = re.search(r"-(\d+)$", question_id or "")
        if match:
            return int(match.group(1)), str(question_id)
        return 10**9, str(question_id)

    def _compute_group_ratings(self, items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            meta = item.get("pp_meta", {})
            video_id = str(meta.get("gt_video") or item.get("gt_video") or "").strip()
            if not video_id:
                continue
            grouped.setdefault(video_id, []).append(item)

        total_scores: List[float] = []
        relevance_scores: List[float] = []
        relevance_linear_scores: List[float] = []
        logic_scores: List[float] = []
        level_scores: Dict[str, List[float]] = {}
        second_head_scores: Dict[str, List[float]] = {}
        third_head_scores: Dict[str, List[float]] = {}

        for _, group_items in sorted(grouped.items()):
            ordered_items = sorted(
                group_items,
                key=lambda item: self._group_question_order(item.get("pp_meta", {}).get("gt_question_id", "")),
            )
            meta = ordered_items[-1].get("pp_meta", {})
            scores = []
            for item in ordered_items:
                pred = self.parse_answer(item.get("response", ""), candidates=item.get("pp_meta", {}).get("gt_candidates"))
                gold = str(item.get("pp_meta", {}).get("gt_answer") or "").strip()
                scores.append(1 if pred and gold and pred == gold else 0)

            if len(scores) != 4:
                continue

            group_type = str(meta.get("gt_group_type") or "").strip().lower()
            if group_type == "relevance":
                exp_score, linear_score = _get_relevance_scores(scores)
                relevance_scores.append(exp_score)
                relevance_linear_scores.append(linear_score)
                group_score = exp_score
            elif group_type == "logic":
                group_score = _get_logic_score(scores, str(meta.get("gt_group_structure") or ""))
                logic_scores.append(group_score)
            else:
                logger.warning(f"[Open-o3 Eval] Skip unknown VideoMME-v2 group_type: {group_type!r}")
                continue

            total_scores.append(group_score)

            level = str(meta.get("gt_level") or "").strip()
            if level:
                level_scores.setdefault(level, []).append(group_score)

            second_head = str(meta.get("gt_second_head") or "").strip()
            if second_head:
                second_head_scores.setdefault(second_head, []).append(group_score)

            third_head = str(meta.get("gt_third_head") or "").strip()
            if third_head:
                third_head_scores.setdefault(third_head, []).append(group_score)

        def _avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 2) if values else 0.0

        metrics: Dict[str, Dict[str, float]] = {
            "final_rating": {
                "total": _avg(total_scores),
                "relevance_score": _avg(relevance_scores),
                "relevance_linear_score": _avg(relevance_linear_scores),
                "logic_score": _avg(logic_scores),
            }
        }

        for level, values in sorted(level_scores.items()):
            metrics[f"level/{level}"] = {"score": _avg(values)}
        for second_head, values in sorted(second_head_scores.items()):
            metrics[f"second_head/{second_head}"] = {"score": _avg(values)}
        for third_head, values in sorted(third_head_scores.items()):
            metrics[f"third_head/{third_head}"] = {"score": _avg(values)}

        return metrics

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(data_list)
        valid = 0
        correct = 0
        for item in data_list:
            meta = item.get("pp_meta", {})
            candidates = meta.get("gt_candidates") or item.get("gt_candidates") or []
            pred = self.parse_answer(item.get("response", ""), candidates=candidates)
            gold = str(meta.get("gt_answer") or item.get("gt_answer") or "").strip()
            if pred:
                valid += 1
            if pred and gold and pred == gold:
                correct += 1

        metrics: Dict[str, Any] = {
            "overall": {
                "total": total,
                "valid": valid,
                "acc": round(correct * 100 / total, 2) if total > 0 else 0.0,
                "valid_acc": round(correct * 100 / valid, 2) if valid > 0 else 0.0,
            }
        }
        metrics.update(self._compute_group_ratings(data_list))
        return metrics

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        meta = result.get("pp_meta", {})
        candidates = meta.get("gt_candidates") or result.get("gt_candidates") or []
        pred = self.parse_answer(result.get("response", ""), candidates=candidates)
        gold = str(meta.get("gt_answer") or result.get("gt_answer") or "").strip().upper()
        is_valid = bool(pred)
        is_correct = bool(pred and gold and pred == gold)
        return {
            "sample_eval": {
                "metric_type": "classification",
                "pred": pred,
                "gold": gold,
                "is_valid": is_valid,
                "is_correct": is_correct,
                "acc": float(is_correct),
                "group_type": str(meta.get("gt_group_type") or "").strip().lower(),
                "group_structure": str(meta.get("gt_group_structure") or "").strip(),
            }
        }
