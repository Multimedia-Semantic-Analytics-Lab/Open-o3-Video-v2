import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

import openo3_eval.env as env
from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY, AnswerParsers, DataSets
from openo3_eval.runtime import get_logger
from openo3_eval.video_o3.message import build_video_o3_custom_message
from openo3_eval.video_o3.prompts import (
    QUESTION_TEMPLATE_MC,
    QUESTION_TEMPLATE_OE,
    build_direct_answer_prompt,
    build_direct_answer_prompt_oe,
    build_think_answer_prompt,
    build_think_answer_prompt_oe,
)
from openo3_eval.vl_utils.extract_frames import batch_extract_frames

logger = get_logger()

VIDEOMMMU_CATEGORIES = ("Perception", "Comprehension", "Adaptation")
CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_VIDEOMMMU_QUESTION_IMAGE_DIR = os.environ.get("VIDEOMMMU_IMAGE_DIR", "data/benchmarks/VideoMMMU/images")
QUESTION_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
SUBJECT_GROUPS = {
    "Art": "Art and Design",
    "Art_Theory": "Art and Design",
    "Design": "Art and Design",
    "Music": "Art and Design",
    "Accounting": "Business",
    "Economics": "Business",
    "Finance": "Business",
    "Manage": "Business",
    "Marketing": "Business",
    "Biology": "Science",
    "Chemistry": "Science",
    "Geography": "Science",
    "Math": "Science",
    "Physics": "Science",
    "History": "Humanities and Social Science",
    "Literature": "Humanities and Social Science",
    "Sociology": "Humanities and Social Science",
    "Psychology": "Humanities and Social Science",
    "Agriculture": "Engineering",
    "Architecture_and_Engineering": "Engineering",
    "Computer_Science": "Engineering",
    "Electronics": "Engineering",
    "Energy_and_Power": "Engineering",
    "Materials": "Engineering",
    "Mechanical_Engineering": "Engineering",
    "Basic_Medical_Science": "Health and Medicine",
    "Clinical_Medicine": "Health and Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Health and Medicine",
    "Pharmacy": "Health and Medicine",
    "Public_Health": "Health and Medicine",
}
MEDIA_ROOT_BY_SUBJECT = {
    "Art": "Art",
    "Art_Theory": "Art",
    "Design": "Art",
    "Music": "Art",
    "Biology": "Science",
    "Chemistry": "Science",
    "Geography": "Science",
    "Math": "Science",
    "Physics": "Science",
    "History": "Humanities",
    "Literature": "Humanities",
    "Sociology": "Humanities",
    "Psychology": "Humanities",
    "Agriculture": "Engineering",
    "Architecture_and_Engineering": "Engineering",
    "Computer_Science": "Engineering",
    "Electronics": "Engineering",
    "Energy_and_Power": "Engineering",
    "Materials": "Engineering",
    "Mechanical_Engineering": "Engineering",
    "Basic_Medical_Science": "Medicine",
    "Clinical_Medicine": "Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Medicine",
    "Pharmacy": "Medicine",
    "Public_Health": "Medicine",
    "Accounting": "Business",
    "Economics": "Business",
    "Finance": "Business",
    "Manage": "Business",
    "Marketing": "Business",
}


def _extract_subject(sample_id: str) -> str:
    parts = str(sample_id or "").split("_")
    if len(parts) < 3:
        return ""
    return "_".join(parts[1:-1])


def _extract_subdomain(sample_id: str) -> str:
    match = re.search(r"^[^_]+_(.+?)_\d+$", str(sample_id or ""))
    return match.group(1) if match else _extract_subject(sample_id)


def _normalize_question_type(raw_value: Any, has_options: bool) -> str:
    text = str(raw_value or "").strip().lower()
    if text in {"multiple-choice", "multiple_choice", "mc"}:
        return "multiple-choice"
    if text in {"open", "oe"}:
        return "open"
    return "multiple-choice" if has_options else "open"


def _coerce_option_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if hasattr(raw_value, "tolist"):
        raw_value = raw_value.tolist()
    if isinstance(raw_value, tuple):
        raw_value = list(raw_value)
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    raw_text = str(raw_value).strip()
    if not raw_text or raw_text == "[]":
        return []

    try:
        parsed = ast.literal_eval(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    quoted_values = [left or right for left, right in re.findall(r"'([^']*)'|\"([^\"]*)\"", raw_text)]
    if quoted_values:
        return [value.strip() for value in quoted_values if value.strip()]

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines


def _format_choices(options: Iterable[str]) -> List[str]:
    formatted: List[str] = []
    for idx, option in enumerate(options):
        option_text = str(option).strip()
        if not option_text:
            continue
        letter = CHOICE_LETTERS[idx]
        if re.match(rf"^{letter}\s*[\.\):]", option_text, flags=re.IGNORECASE):
            formatted.append(option_text)
        else:
            formatted.append(f"{letter}. {option_text}")
    return formatted


def _resolve_video_path(sample_id: str, benchmark_root: Path, row: Dict[str, Any]) -> str:
    explicit = str(row.get("video_path") or "").strip()
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.is_absolute() and explicit_path.exists():
            return str(explicit_path)
        joined = benchmark_root / explicit
        if joined.exists():
            return str(joined)

    subject = _extract_subject(sample_id)
    media_root = MEDIA_ROOT_BY_SUBJECT.get(subject, "")
    candidates = []
    if media_root:
        candidates.append(benchmark_root / media_root / f"{sample_id}.mp4")
    candidates.append(benchmark_root / f"{sample_id}.mp4")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def _resolve_question_image_candidate(candidate: str, image_dir: Path) -> str:
    candidate_text = str(candidate or "").strip()
    if not candidate_text:
        return ""

    candidate_path = Path(candidate_text)
    if candidate_path.exists():
        return str(candidate_path)

    joined = image_dir / candidate_path.name
    if joined.exists():
        return str(joined)
    return ""


def _question_image_from_sample_id(sample_id: str, image_dir: Path) -> str:
    sample_id = str(sample_id or "").strip()
    if not sample_id:
        return ""
    for suffix in QUESTION_IMAGE_EXTENSIONS:
        candidate = image_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    return ""


def _ensure_question_image_path(row: Dict[str, Any], image_dir: Path, cache_dir: Path) -> str:
    explicit = str(row.get("question_image_path") or "").strip()
    resolved_explicit = _resolve_question_image_candidate(explicit, image_dir)
    if resolved_explicit:
        return resolved_explicit

    image_value = row.get("image")
    image_file_name = ""
    if isinstance(image_value, dict):
        image_file_name = str(image_value.get("path") or "").strip()
    resolved_named = _resolve_question_image_candidate(image_file_name, image_dir)
    if resolved_named:
        return resolved_named

    sample_id = str(row.get("id") or "").strip()
    resolved_from_id = _question_image_from_sample_id(sample_id, image_dir)
    if resolved_from_id:
        return resolved_from_id

    image_value = row.get("image")
    if not isinstance(image_value, dict):
        return ""

    image_bytes = image_value.get("bytes")
    if not isinstance(image_bytes, (bytes, bytearray)):
        return ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(image_file_name).name if image_file_name else f"{sample_id or 'question_image'}.png"
    output_path = cache_dir / file_name
    if not output_path.exists():
        output_path.write_bytes(bytes(image_bytes))
    return str(output_path)


def _load_videommmu_rows(dataset_path: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
    if dataset_path.is_file():
        if dataset_path.suffix == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError(f"[Open-o3 Eval] VideoMMMU json must contain a list of rows: {dataset_path}")
            return rows, dataset_path, dataset_path.parent
        raise ValueError(f"[Open-o3 Eval] Unsupported VideoMMMU annotation file: {dataset_path}")

    rows: List[Dict[str, Any]] = []
    for category in VIDEOMMMU_CATEGORIES:
        parquet_path = dataset_path / category / "test-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"[Open-o3 Eval] VideoMMMU split not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        for row in df.to_dict(orient="records"):
            row["category"] = category
            rows.append(row)
    return rows, dataset_path, dataset_path


def _normalize_open_gold(raw_value: Any) -> List[Any]:
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if isinstance(raw_value, list):
        return raw_value

    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []

    try:
        parsed = ast.literal_eval(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, list):
        return parsed
    quoted_values = [left or right for left, right in re.findall(r"'([^']*)'|\"([^\"]*)\"", raw_text)]
    if quoted_values:
        return [value.strip() for value in quoted_values if value.strip()]
    return [raw_text]


def _normalize_open_candidate(value: Any) -> List[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        number = round(float(text.replace(",", "")), 2)
        return [number]
    except Exception:
        pass

    lowered = text.lower()
    if len(lowered) == 1:
        return [f" {lowered}", f"{lowered} "]
    return [lowered]


def _open_answer_is_correct(gold: Any, pred: Any) -> bool:
    gold_values = gold if isinstance(gold, list) else [gold]
    normalized_gold: List[Any] = []
    for value in gold_values:
        normalized_gold.extend(_normalize_open_candidate(value))

    pred_values = pred if isinstance(pred, list) else [pred]
    for pred_value in pred_values:
        if isinstance(pred_value, str):
            pred_text = pred_value.lower()
            for gold_value in normalized_gold:
                if isinstance(gold_value, str) and gold_value in pred_text:
                    return True
        else:
            if pred_value in normalized_gold:
                return True
    return False


def _summarize_group(items: List[Dict[str, Any]], dataset: "VideoMMMU") -> Dict[str, Any]:
    total = len(items)
    valid = 0
    correct = 0
    for item in items:
        pp_meta = item.get("pp_meta", {})
        question_type = str(pp_meta.get("gt_question_type") or "multiple-choice")
        pred = dataset.parse_answer(
            item.get("response", ""),
            question_type=question_type,
            candidates=pp_meta.get("gt_candidates") or [],
        )
        if question_type == "multiple-choice":
            gold = str(pp_meta.get("gt_answer") or "").strip().upper()
            is_valid = bool(pred)
            is_correct = bool(pred and gold and pred == gold)
        else:
            gold = pp_meta.get("gt_answer_open") or _normalize_open_gold(pp_meta.get("gt_answer"))
            is_valid = bool(pred)
            is_correct = _open_answer_is_correct(gold, pred) if pred else False
        valid += int(is_valid)
        correct += int(is_correct)

    return {
        "total": total,
        "valid": valid,
        "valid_pct": round(valid * 100 / total, 2) if total > 0 else 0.0,
        "correct": correct,
        "acc": round(correct * 100 / total, 2) if total > 0 else 0.0,
    }


@DATASET_REGISTRY.register(name=DataSets.videommmu)
class VideoMMMU(BaseDatasetPP):
    user_prompt_default = Path(__file__).resolve().parents[1] / "prompts" / "videommmu_mc.txt"
    dataset_name = DataSets.videommmu
    answer_parser = AnswerParsers.videommmu_parser

    dataset_dir_env_key = "VIDEOMMMU_DATASET_DIR"
    user_prompt_env_key = "VIDEOMMMU_USER_PROMPT"
    input_mode_env_key = "VIDEOMMMU_INPUT_MODE"
    sample_fps_env_key = "VIDEOMMMU_SAMPLE_FPS"
    resize_env_key = "VIDEOMMMU_RESIZE"

    @staticmethod
    def _question_image_block(question_image_path: str) -> str:
        return "<image>\n" if question_image_path else ""

    @staticmethod
    def _category_instruction(category: str, question_type: str) -> str:
        if category in {"Perception", "Comprehension"}:
            return "Please ignore the quiz question that may appear in the last frame of the video."
        if category == "Adaptation" and question_type == "open":
            return "You should learn the video content and use the reference image when answering the question."
        if category == "Adaptation":
            return "You should learn the video content and use the reference image if it is provided."
        return ""

    def _build_prompt_text(
        self,
        *,
        question: str,
        category: str,
        question_type: str,
        choice: str,
        question_image_path: str,
        prompt_replacements: Dict[str, str] | None = None,
    ) -> str:
        default_prompt = (
            Path(__file__).resolve().parents[1] / "prompts" / "videommmu_oe.txt"
            if question_type == "open"
            else Path(__file__).resolve().parents[1] / "prompts" / "videommmu_mc.txt"
        )
        return self.build_user_prompt(
            default=default_prompt,
            prompt_replacements=prompt_replacements,
            question=question,
            category=category,
            category_instruction=self._category_instruction(category, question_type),
            choice=choice,
            question_image_block=self._question_image_block(question_image_path),
            question_image_suffix=" and the reference image" if question_image_path else "",
        )

    @staticmethod
    def _build_video_o3_context_lines(category: str, question_type: str, has_question_image: bool) -> List[str]:
        instruction = VideoMMMU._category_instruction(category, question_type)
        lines = [f"Question Category: {category}"]
        if instruction:
            lines.append(instruction)
        if has_question_image:
            lines.append("A reference image related to the question is also provided.")
        return lines

    @classmethod
    def _build_video_o3_prompt(
        cls,
        question: str,
        category: str,
        question_type: str,
        choice: str,
        *,
        has_question_image: bool,
    ) -> str:
        lines = cls._build_video_o3_context_lines(category, question_type, has_question_image)
        if question_type == "open":
            lines.append(QUESTION_TEMPLATE_OE.format(question=question).strip())
            return "\n".join(line for line in lines if line.strip())

        lines.append(QUESTION_TEMPLATE_MC.format(question=question, options=choice).strip())
        return "\n".join(line for line in lines if line.strip())

    @classmethod
    def _build_video_o3_direct_prompt(
        cls,
        question: str,
        category: str,
        question_type: str,
        choice: str,
        *,
        has_question_image: bool,
    ) -> str:
        lines = cls._build_video_o3_context_lines(category, question_type, has_question_image)
        if question_type == "open":
            lines.append(build_direct_answer_prompt_oe(question).strip())
            return "\n".join(line for line in lines if line.strip())

        lines.append(build_direct_answer_prompt(question, choice).strip())
        return "\n".join(line for line in lines if line.strip())

    @classmethod
    def _build_video_o3_think_prompt(
        cls,
        question: str,
        category: str,
        question_type: str,
        choice: str,
        *,
        has_question_image: bool,
    ) -> str:
        lines = cls._build_video_o3_context_lines(category, question_type, has_question_image)
        if question_type == "open":
            lines.append(build_think_answer_prompt_oe(question).strip())
            return "\n".join(line for line in lines if line.strip())

        lines.append(build_think_answer_prompt(question, choice).strip())
        return "\n".join(line for line in lines if line.strip())

    @staticmethod
    def _inject_question_image(message: List[Dict[str, Any]], question_image_path: str) -> List[Dict[str, Any]]:
        if not question_image_path:
            return message
        if not message or message[-1].get("type") != "text":
            return message
        return [
            *message[:-1],
            {"type": "text", "value": "Question reference image:\n<image>\n"},
            {"type": "image", "value": question_image_path},
            message[-1],
        ]

    def build_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = self.get_dataset_dir()
        input_mode = self.get_input_mode()
        sample_fps = self.get_sample_fps()
        resize = self.get_resize()
        frame_cache_dir = self.get_frame_cache_dir(self.dataset_name)
        question_image_cache_dir = Path(
            os.environ.get("VIDEOMMMU_QUESTION_IMAGE_DIR", DEFAULT_VIDEOMMMU_QUESTION_IMAGE_DIR)
        )
        question_image_materialize_dir = Path(env.CACHE_DIR) / self.dataset_name.value / "question_images"

        self.dataset = []
        rows, anno_path, benchmark_root = _load_videommmu_rows(dataset_path)

        all_items: List[Tuple[Dict[str, Any], str, str, str, str, str, str, List[str], str]] = []
        skipped_missing_video = 0
        for row in rows:
            sample_id = str(row.get("id") or "").strip()
            options = _coerce_option_list(row.get("options"))
            question_type = _normalize_question_type(row.get("question_type"), has_options=bool(options))
            choice_lines = _format_choices(options)
            choice_text = "\n".join(choice_lines)
            video_path = _resolve_video_path(sample_id, benchmark_root, row)
            if not video_path:
                skipped_missing_video += 1
                continue

            question_image_path = _ensure_question_image_path(
                row,
                question_image_cache_dir,
                question_image_materialize_dir,
            )
            subject = _extract_subject(sample_id)
            domain_group = SUBJECT_GROUPS.get(subject, "unknown")
            category = str(row.get("category") or "").strip()
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()

            all_items.append(
                (
                    row,
                    sample_id,
                    video_path,
                    question_image_path,
                    subject,
                    domain_group,
                    question_type,
                    choice_lines,
                    choice_text,
                )
            )

        cache_results: Dict[str, Tuple[List[str], List[float]]] = {}
        if input_mode == "image":
            unique_videos = list({video_path for _, _, video_path, _, _, _, _, _, _ in all_items})
            cache_results = batch_extract_frames(
                video_paths=unique_videos,
                cache_dir=frame_cache_dir,
                sample_fps=sample_fps,
                resize=resize,
            )

        skipped_frames = 0
        for row, sample_id, video_path, question_image_path, subject, domain_group, question_type, choice_lines, choice_text in all_items:
            category = str(row.get("category") or "").strip()
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            pp_meta = {
                "gt_answer": answer,
                "gt_answer_open": _normalize_open_gold(answer) if question_type == "open" else [],
                "gt_candidates": choice_lines,
                "gt_question": question,
                "gt_options_text": choice_text,
                "gt_id": sample_id,
                "gt_category": category,
                "gt_subject": subject,
                "gt_domain_group": domain_group,
                "gt_question_type": question_type,
                "gt_qa_type": str(row.get("qa_type") or "").strip(),
                "gt_link_selected": str(row.get("link_selected") or "").strip(),
                "gt_question_image_path": question_image_path,
            }

            if self.is_video_path_input_mode(input_mode):
                prompt = self._build_prompt_text(
                    question=question,
                    category=category,
                    question_type=question_type,
                    choice=choice_text,
                    question_image_path=question_image_path,
                )
                item = {
                    "messages": [{"role": "user", "content": prompt}],
                    "videos": [video_path],
                    "pp_meta": pp_meta,
                }
                if question_image_path:
                    item["images"] = [question_image_path]
                self.dataset.append(item)
                continue

            cached = cache_results.get(video_path)
            if not cached:
                skipped_frames += 1
                continue
            frame_paths, frame_ts = cached
            image_token_content = "".join(f"<{ts:.1f} seconds><image>" for ts in frame_ts)
            prompt = self._build_prompt_text(
                question=question,
                category=category,
                question_type=question_type,
                choice=choice_text,
                question_image_path=question_image_path,
                prompt_replacements={"<video>": image_token_content},
            )
            images = frame_paths + ([question_image_path] if question_image_path else [])
            item = {
                "messages": [{"role": "user", "content": prompt}],
                "images": images,
                "pp_meta": pp_meta,
            }
            self.dataset.append(item)

        logger.info(
            f"[Open-o3 Eval] VideoMMMU loaded {len(self.dataset)} samples from {anno_path}; "
            f"skipped {skipped_missing_video} missing videos, {skipped_frames} frame failures (mode={input_mode})"
        )
        return self.dataset

    def build_video_o3_tasks(self, args: Any, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        pp_meta = item.get("pp_meta", {})
        video_path = str((item.get("videos") or [""])[0])
        question = str(pp_meta.get("gt_question") or "").strip()
        category = str(pp_meta.get("gt_category") or "").strip()
        question_type = str(pp_meta.get("gt_question_type") or "multiple-choice")
        choice = str(pp_meta.get("gt_options_text") or "").strip()
        question_image_path = str(pp_meta.get("gt_question_image_path") or "").strip()
        has_question_image = bool(question_image_path)

        message, frame_meta = build_video_o3_custom_message(
            args,
            video_path=video_path,
            prompt_text=self._build_video_o3_prompt(
                question, category, question_type, choice, has_question_image=has_question_image
            ),
            dataset_name=self.dataset_name.value if hasattr(self.dataset_name, "value") else str(self.dataset_name),
            question_type="oe" if question_type == "open" else "mc",
            direct_prompt=self._build_video_o3_direct_prompt(
                question, category, question_type, choice, has_question_image=has_question_image
            ),
            think_answer_prompt=self._build_video_o3_think_prompt(
                question, category, question_type, choice, has_question_image=has_question_image
            ),
        )
        message = self._inject_question_image(message, question_image_path)
        return [{"task_name": "default", "message": message, "frame_meta": frame_meta, "skip": False}]

    def calc_metric(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"overall": _summarize_group(data_list, self)}

        for dim_key, meta_key in (
            ("category", "gt_category"),
            ("subject", "gt_subject"),
            ("domain_group", "gt_domain_group"),
            ("question_type", "gt_question_type"),
            ("qa_type", "gt_qa_type"),
        ):
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for item in data_list:
                value = str(item.get("pp_meta", {}).get(meta_key) or "").strip()
                if value:
                    grouped.setdefault(value, []).append(item)
            for group_name, items in sorted(grouped.items()):
                metrics[f"{dim_key}/{group_name}"] = _summarize_group(items, self)

        return metrics

    def build_sample_eval(self, item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        pp_meta = result.get("pp_meta", {})
        question_type = str(pp_meta.get("gt_question_type") or "multiple-choice")
        pred = self.parse_answer(
            result.get("response", ""),
            question_type=question_type,
            candidates=pp_meta.get("gt_candidates") or [],
        )
        if question_type == "multiple-choice":
            gold = str(pp_meta.get("gt_answer") or "").strip().upper()
            correct = bool(pred and gold and pred == gold)
        else:
            gold = pp_meta.get("gt_answer_open") or _normalize_open_gold(pp_meta.get("gt_answer"))
            correct = _open_answer_is_correct(gold, pred) if pred else False

        return {
            "sample_eval": {
                "metric_type": "open_ended_accuracy" if question_type == "open" else "multiple_choice_accuracy",
                "pred_answer": pred,
                "gold_answer": pp_meta.get("gt_answer"),
                "correct": correct,
            }
        }
