from argparse import Namespace
from pathlib import Path

import openo3_eval.parser  # noqa: F401
from openo3_eval.datasetpp.longvideobench import LongVideoBench, _extract_answer_letter
from openo3_eval.datasetpp.lvbench import LVBench, _extract_question_and_candidates


def test_lvbench_extract_question_and_candidates() -> None:
    question, candidates = _extract_question_and_candidates(
        "What color is the car?\n(A) Red\n(B) Blue\n(C) Green\n(D) Black"
    )

    assert question == "What color is the car?"
    assert candidates == ["Red", "Blue", "Green", "Black"]


def test_lvbench_build_dataset_from_json(tmp_path: Path) -> None:
    root = tmp_path / "LVBench"
    videos = root / "video_chunks"
    videos.mkdir(parents=True)
    (videos / "abc.mp4").write_bytes(b"")
    anno = root / "lvbench.json"
    anno.write_text(
        '[{"id":"1","video_id":"abc","video_type":"cartoon","video_rel_path":"video_chunks/abc.mp4",'
        '"question":"What color is the car?","raw_question":"What color is the car?\\n(A) Red\\n(B) Blue",'
        '"candidates":["Red","Blue"],"answer":"B","question_types":["entity recognition"],'
        '"time_reference":"00:01-00:02"}]',
        encoding="utf-8",
    )

    bench = LVBench(Namespace(dataset_dir=str(anno), answer_parser="", result_path=None))
    dataset = bench.build_dataset()

    assert dataset
    sample = dataset[0]
    assert sample["videos"] == [str(videos / "abc.mp4")]
    assert sample["pp_meta"]["gt_answer"] == "B"
    assert sample["pp_meta"]["gt_candidates"] == ["Red", "Blue"]
    assert "A. Red" in sample["pp_meta"]["gt_options_text"]


def test_lvbench_calc_metric() -> None:
    bench = LVBench(Namespace(dataset_dir="", answer_parser="", result_path=None))
    rows = [
        {
            "pp_meta": {
                "gt_answer": "B",
                "gt_candidates": ["Apple", "Banana", "Orange", "Pear"],
                "gt_video_type": "cartoon",
                "gt_question_types": ["entity recognition"],
            },
            "response": "<answer>B</answer>",
        }
    ]

    metrics = bench.calc_metric(rows)

    assert metrics["overall"]["acc"] == 100.0
    assert metrics["video_type/cartoon"]["acc"] == 100.0
    assert metrics["question_type/entity recognition"]["acc"] == 100.0


def test_longvideobench_build_dataset_from_json(tmp_path: Path) -> None:
    root = tmp_path / "LongVideoBench"
    videos = root / "videos"
    videos.mkdir(parents=True)
    (videos / "abc.mp4").write_bytes(b"")
    anno = root / "longvideobench.json"
    anno.write_text(
        '[{"id":"1","video_id":"abc","video_path":"videos/abc.mp4",'
        '"question":"What happens at the end?","candidates":["Run","Stop","Jump"],'
        '"correct_choice":1,"duration_group":"long","question_category":"plot",'
        '"topic_category":"daily","level":"easy"}]',
        encoding="utf-8",
    )

    bench = LongVideoBench(Namespace(dataset_dir=str(anno), answer_parser="", result_path=None))
    dataset = bench.build_dataset()

    assert dataset
    sample = dataset[0]
    assert sample["videos"] == [str(videos / "abc.mp4")]
    assert sample["pp_meta"]["gt_answer"] == "B"
    assert sample["pp_meta"]["gt_candidates"] == ["Run", "Stop", "Jump"]
    assert "B. Stop" in sample["pp_meta"]["gt_options_text"]


def test_longvideobench_answer_falls_back_to_correct_choice() -> None:
    row = {"answer": "Stop", "correct_choice": 1}

    assert _extract_answer_letter(row, 3) == "B"


def test_longvideobench_calc_metric() -> None:
    bench = LongVideoBench(Namespace(dataset_dir="", answer_parser="", result_path=None))
    rows = [
        {
            "pp_meta": {
                "gt_answer": "B",
                "gt_candidates": ["Run", "Stop", "Jump"],
                "gt_duration_group": "long",
                "gt_question_category": "plot",
                "gt_topic_category": "daily",
                "gt_level": "easy",
            },
            "response": "The best answer is B.",
        }
    ]

    metrics = bench.calc_metric(rows)

    assert metrics["overall"]["acc"] == 100.0
    assert metrics["duration_group/long"]["acc"] == 100.0
    assert metrics["question_category/plot"]["acc"] == 100.0
    assert metrics["topic_category/daily"]["acc"] == 100.0
    assert metrics["level/easy"]["acc"] == 100.0
