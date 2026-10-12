"""Convert released annotations into timestamped MS-SWIFT multi-image JSONL.

The source JSON contains STGR records and multi-observation trajectories.
Videos are sampled into images with explicit source-video timestamps so crop
observations and key frames retain their temporal grounding.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


SOURCES = {
    "stgr_sft_version_0501_18k": "STGR-SFT-Refine-0501-18K.json",
    "stgr_timelens_32k_temporal_qa": "timelens_32k_stgr_style_temporal_qa.json",
    "seeker_sft_w_tool_mix_grounded_version_0501_52k_multi_image": "video-o3_with_grounded_cot-Rewrite-0613.json",
    "seeker_sft_cgbench_multi_image_obs1fps_crop124fps_cap96": "cgbench_gemini_cot_results_5301_rewrite_0613.json",
    "seeker_sft_longvt_multi_image_obs1fps_crop124fps_cap96": "sft_longvt_longvideo_reflection_2000.json",
    "stgr_caprl_caption_12k_obs1fps_cap96": "caprl_caption_2_3_min_stgr.json",
    "stgr_omni100k_mcq_5_7k_obs1fps_cap96": "omni100k_mcq_5_7k_wcot.json",
    "stgr_omni100k_oe_7k_obs1fps_cap96": "omni100k_oe_7k_wcot.json",
}

STGR_SYSTEM = {
    "visual QA": (
        "Answer the image question. Put reasoning inside <think> tags and the final answer "
        "inside <answer> tags. Ground relevant objects with <obj> and <box> tags."
    ),
    "temporal QA": (
        "Find the video interval answering the question. Put reasoning inside <think> tags. "
        "Return <answer>From <t>start_time</t>s to <t>end_time</t>s</answer>."
    ),
    "temporal QA (MCQ)": (
        "Find the relevant video interval and answer the multiple-choice question. "
        "Put reasoning inside <think> tags and the final answer inside <answer> tags."
    ),
    "temporal-spatial free-form QA": (
        "Answer based on video evidence. Put reasoning inside <think> tags and the final answer "
        "inside <answer> tags. Ground relevant objects with <obj>, <box>, and <t> tags."
    ),
    "General video QA MCQ": (
        "Answer the multiple-choice video question. Put reasoning inside <think> tags; "
        "put only the option letter inside <answer> tags."
    ),
    "General video QA Free-form": (
        "Answer the video question. Put reasoning inside <think> tags and the final answer "
        "inside <answer> tags. Ground relevant objects and moments in visual evidence."
    ),
}


def resolve_media(raw: str, media_root: Path, manifest: dict[str, str]) -> Path:
    if raw in manifest:
        path = Path(manifest[raw]).expanduser().resolve()
    else:
        relative = Path(raw)
        if relative.is_absolute():
            path = relative
        else:
            candidates = [media_root / relative]
            candidates += [media_root / part / relative for part in (
                "videor1", "timerft", "timelens", "seeker", "cgbench", "longvt",
                "caprl", "omni100k", "treevgr", "gqa", "videoespresso/videos", "tvg_r1",
                "stgr/plm/videos", "stgr/plm/kfs",
                "stgr/temporal_grounding/videos", "stgr/temporal_grounding/kfs",
            )]
            path = next((item for item in candidates if item.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(f"Missing media: {raw} (resolved to {path})")
    return path.resolve()


def sample_frames(video: Path, crop: list[float], sample: str, cache_root: Path, max_frames: int) -> list[str]:
    fps = {"full": 1, "coarse": 1, "medium": 2, "fine": 4}.get(sample)
    if fps is None:
        raise ValueError(f"Unknown sampling strategy: {sample}")
    key = hashlib.sha256(f"{video}|{crop}|{sample}|{max_frames}".encode()).hexdigest()[:24]
    target = cache_root / key
    complete = target / ".complete"
    frames = sorted(target.glob("*.jpg")) if complete.is_file() else []
    if not frames:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to extract training frames")
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("*.jpg"):
            stale.unlink()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if crop and len(crop) == 2 and crop[0] >= 0 and crop[1] > crop[0]:
            command += ["-ss", str(crop[0]), "-t", str(crop[1] - crop[0])]
        command += ["-i", str(video), "-vf", f"fps={fps},scale=-2:224", "-frames:v", str(max_frames),
                    str(target / "%04d.jpg")]
        subprocess.run(command, check=True)
        frames = sorted(target.glob("*.jpg"))
        if frames:
            complete.touch()
    if not frames:
        raise ValueError(f"No frames extracted from {video} with crop={crop}")
    return [str(frame.resolve()) for frame in frames]


def timed_frames(frames: list[str], crop: list[float], sample: str) -> list[tuple[float, str]]:
    fps = {"full": 1, "coarse": 1, "medium": 2, "fine": 4}[sample]
    start = float(crop[0]) if crop and crop[0] >= 0 else 0.0
    return [(start + index / fps, frame) for index, frame in enumerate(frames)]


def frame_prompt(frames: list[tuple[float, str]]) -> str:
    return "\n".join(f"<t>{timestamp:g}</t>s: <image>" for timestamp, _ in frames)


def convert_stgr(row: dict[str, Any], media_root: Path, manifest: dict[str, str],
                 cache_root: Path, max_frames: int, check_only: bool) -> dict[str, Any]:
    task = str(row.get("task", ""))
    if task not in STGR_SYSTEM:
        raise ValueError(f"Unknown STGR task: {task}")
    system = STGR_SYSTEM[task]
    question = str(row["question"])
    reasoning = str(row.get("reasoning_process") or "").strip()
    answer = row.get("answer", "")
    if isinstance(answer, list) and len(answer) == 2:
        answer = f"From <t>{answer[0]}</t>s to <t>{answer[1]}</t>s"
    response = f"<think>{reasoning}</think>\n<answer>{answer}</answer>"
    image_path = row.get("image_path")
    video_path = row.get("video_path")
    if image_path:
        image = resolve_media(str(image_path), media_root, manifest)
        return {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": response},
        ], "images": [str(image)]}
    if not video_path:
        raise ValueError(f"STGR row {row.get('id')} has no video_path or image_path")
    video = resolve_media(str(video_path), media_root, manifest)
    frames = [] if check_only else timed_frames(
        sample_frames(video, [-1, -1], "full", cache_root, max_frames), [-1, -1], "full"
    )
    key_frames = row.get("key_frames") or []
    if key_frames:
        extras = [(float(item["time"]), resolve_media(str(item["path"]), media_root, manifest))
                  for item in key_frames]
        if not check_only:
            frames += [(timestamp, str(path)) for timestamp, path in extras]
            frames.sort(key=lambda item: item[0])
    frame_text = frame_prompt(frames)
    content = f"Video observations:\n{frame_text}\n{question}" if frame_text else question
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
        {"role": "assistant", "content": response},
    ], "images": [path for _, path in frames]}


def convert_trajectory(row: dict[str, Any], media_root: Path, manifest: dict[str, str],
                       cache_root: Path, max_frames: int, check_only: bool) -> dict[str, Any]:
    messages = row["messages"]
    observations = row["videos"]
    count = sum(str(message.get("content", "")).count("<video>") for message in messages)
    if count != len(observations):
        raise ValueError(f"Row {row.get('id')} has {count} <video> markers and {len(observations)} observations")
    images = []
    replacements = []
    for observation in observations:
        video = resolve_media(str(observation["url"]), media_root, manifest)
        crop = observation.get("crop", [-1, -1])
        sample = observation.get("sample", "full")
        frames = [] if check_only else timed_frames(
            sample_frames(video, crop, sample, cache_root, max_frames), crop, sample
        )
        replacements.append(frame_prompt(frames) if frames else "[video validated]")
        images.extend(path for _, path in frames)
    replacement_iter = iter(replacements)
    converted_messages = []
    for message in messages:
        content = str(message.get("content", ""))
        while "<video>" in content:
            content = content.replace("<video>", next(replacement_iter), 1)
        converted_messages.append({**message, "content": content})
    return {"messages": converted_messages, "images": images}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Downloaded ModelScope dataset root")
    parser.add_argument("--media-root", type=Path, required=True, help="Root containing downloaded source media")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    parser.add_argument("--media-manifest", type=Path, help="Optional JSON mapping annotation paths to media files")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--check-only", action="store_true", help="Validate media paths without extracting frames")
    parser.add_argument("--sources", nargs="*", choices=sorted(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    manifest = {}
    if args.media_manifest:
        manifest_file = args.media_manifest.expanduser().resolve()
        entries = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(entries, dict):
            parser.error("--media-manifest must contain a JSON object")
        manifest = {
            str(source): str((manifest_file.parent / Path(target).expanduser()).resolve())
            for source, target in entries.items()
        }
    media_root = args.media_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "frames"
    failures = []
    for name in args.sources:
        source = args.data_root / "SFT" / "json_data" / SOURCES[name]
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = json.loads(source.read_text(encoding="utf-8"))
        output = args.output_dir / f"{name}.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        converted = 0
        failed_before = len(failures)
        with (temporary.open("w", encoding="utf-8") if not args.check_only else nullcontext()) as stream:
            for index, row in enumerate(rows):
                try:
                    result = (convert_trajectory(row, media_root, manifest, cache_root, args.max_frames, args.check_only)
                              if "messages" in row else
                              convert_stgr(row, media_root, manifest, cache_root, args.max_frames, args.check_only))
                    if not args.check_only:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    converted += 1
                except (FileNotFoundError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                    failures.append(f"{name}[{index}]: {error}")
        if not args.check_only:
            if len(failures) == failed_before:
                temporary.replace(output)
            else:
                temporary.unlink()
        print(f"{name}: {converted}/{len(rows)} rows {'validated' if args.check_only else 'written'}")
    if failures:
        for failure in failures[:30]:
            print(failure)
        raise SystemExit(f"{len(failures)} rows failed; fix media paths and rerun")


if __name__ == "__main__":
    main()
