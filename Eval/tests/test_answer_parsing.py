import json
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import pytest

import openo3_eval.parser  # noqa: F401
from openo3_eval.datasetpp.base import BaseDatasetPP
from openo3_eval.datasetpp.charades import Charades
from openo3_eval.datasetpp.timelens_bench import TimeLensBench
from openo3_eval.datasetpp.tvgbench import TVGBench
from openo3_eval.datasetpp.video_holmes import VideoHolmes
from openo3_eval.datasetpp.video_mme import VideoMME
from openo3_eval.datasetpp.video_mme_v2 import VideoMMEv2
from openo3_eval.parser import AnswerParsing
from openo3_eval.registry import ANSWER_PARSER_REGISTRY, AnswerParsers, DataSets
from openo3_eval.util import strip_video_o3_metadata
from openo3_eval.video_o3.config import VIDEO_O3_MODEL_CONFIGS
from openo3_eval.video_o3.frames import frame_timestamps_from_meta
from openo3_eval.video_o3.message import build_video_o3_message
from openo3_eval.video_o3.model_multi_image import VideoO3MultiImageChat


class TestAnswerParsing:
    def test_extract_answer_and_ranges(self) -> None:
        parsed = AnswerParsing("prefix <answer>12\u79d2-34\u79d2</answer>").extract_answer_block()
        assert parsed.text == "12\u79d2-34\u79d2"
        assert parsed.parse_time_ranges(allow_hyphen=True, allow_to=False) == [(12.0, 34.0)]

    def test_tracks_truncated_think(self) -> None:
        parsed = AnswerParsing("<think>unfinished").strip_think_blocks()
        assert parsed.is_truncated is True
        assert parsed.text == ""

    def test_parse_time_token_supports_seconds_suffix(self) -> None:
        assert AnswerParsing.parse_time_token("12\u79d2") == 12.0
        assert AnswerParsing.parse_time_token("12.5\u79d2") == 12.5

    def test_parse_time_token_supports_clock_format(self) -> None:
        assert AnswerParsing.parse_time_token("01:23.5") == 83.5

    def test_strip_think_blocks_removes_balanced_block(self) -> None:
        parsed = AnswerParsing("<think>reasoning</think> 12 - 34").strip_think_blocks()
        assert parsed.text == " 12 - 34"
        assert parsed.is_truncated is False

    def test_strip_think_blocks_detects_unclosed_block(self) -> None:
        parsed = AnswerParsing("<think>unfinished").strip_think_blocks()
        assert parsed.text == ""
        assert parsed.is_truncated is True

    def test_strip_think_blocks_detects_stray_closing_tag(self) -> None:
        parsed = AnswerParsing("12 - 34 </think>").strip_think_blocks()
        assert parsed.text == ""
        assert parsed.is_truncated is True

    def test_find_single_angle_second_returns_last_match(self) -> None:
        parsed = AnswerParsing("<1 seconds> x <12.5 seconds>")
        assert parsed.find_single_angle_second() == 12.5


class TestBaseDatasetAnswerParserResolution:
    class DummyDataset(BaseDatasetPP):
        dataset_name = DataSets.timelens_bench
        answer_parser = AnswerParsers.timelens_bench_parser
        input_mode_env_key = "DUMMY_INPUT_MODE"

    def teardown_method(self) -> None:
        ANSWER_PARSER_REGISTRY._obj_map.pop("external_timelens_bench", None)

    def test_prefers_explicit_answer_parser(self) -> None:
        @ANSWER_PARSER_REGISTRY.register(name="external_timelens_bench")
        class ExternalTimeLensBench:
            def __call__(self, _: str, **__: object) -> Tuple[Tuple[float, float], bool]:
                return (90.0, 120.0), False

        dataset = self.DummyDataset(Namespace(dataset_pp=DataSets.timelens_bench, answer_parser="external_timelens_bench"))
        assert dataset.parse_answer("ignored") == ((90.0, 120.0), False)

    def test_falls_back_to_dataset_answer_parser(self) -> None:
        dataset = self.DummyDataset(Namespace())
        assert dataset.parse_answer("12 - 34") == ((12.0, 34.0), False)

    def test_input_mode_uses_video_path_name_and_rejects_legacy_video(self, monkeypatch) -> None:
        dataset = self.DummyDataset(Namespace())
        monkeypatch.setenv("DUMMY_INPUT_MODE", "video_path")
        assert dataset.get_input_mode() == "video_path"

        monkeypatch.setenv("DUMMY_INPUT_MODE", "video")
        with pytest.raises(ValueError, match="must be one of 'image', 'video_path'"):
            dataset.get_input_mode()


class TestTimeLensAnswerParsing:
    def teardown_method(self) -> None:
        ANSWER_PARSER_REGISTRY._obj_map.pop("external_timelens_bench", None)

    def test_parse_module_default_matches_dataset_default(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("12 - 34") == ((12.0, 34.0), False)

    def test_parse_numeric_range(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("12 - 34") == ((12.0, 34.0), False)

    def test_parse_clock_range(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("01:02 - 01:10") == ((62.0, 70.0), False)

    def test_parse_chinese_seconds_range(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("12\u79d2-34\u79d2") == ((12.0, 34.0), False)

    def test_parse_decimal_chinese_seconds_range(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("12.5\u79d2-34.5\u79d2") == ((12.5, 34.5), False)

    def test_strip_balanced_think_and_parse(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("<think>reasoning</think> <answer>12 - 34</answer>") == ((12.0, 34.0), False)

    def test_parse_video_o3_json_result(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer('<answer>{"result": [[12, 34]]}</answer>') == ((12.0, 34.0), False)

    def test_parse_video_o3_temporal_segment(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer('{"temporal_segment": [5, 8]}') == ((5.0, 8.0), False)

    def test_build_sampled_json_file_dataset(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos" / "charades"
        video_dir.mkdir(parents=True)
        (video_dir / "B0XI9.mp4").write_bytes(b"")
        anno_path = tmp_path / "charades-timelens-sampled-131.json"
        anno_path.write_text(
            json.dumps(
                {
                    "B0XI9": {
                        "duration": 31.7,
                        "spans": [[13, 20], [0, 6]],
                        "queries": ["A man is drinking water.", "A person is running."],
                    }
                }
            ),
            encoding="utf-8",
        )
        bench = TimeLensBench(Namespace(dataset_dir=str(anno_path), answer_parser=""))
        rows = bench.build_dataset()
        assert len(rows) == 2
        assert rows[0]["pp_meta"]["gt_dataset"] == DataSets.timelens_bench
        assert rows[0]["pp_meta"]["gt_source_dataset"] == "charades"
        assert rows[0]["pp_meta"]["gt_question"] == "A man is drinking water."

    def test_calc_metric_reports_timelens_overall_and_source(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        metrics = bench.calc_metric(
            [
                {
                    "response": '<answer>{"result": [[10, 20]]}</answer>',
                    "pp_meta": {
                        "gt_dataset": DataSets.timelens_bench,
                        "gt_source_dataset": "charades",
                        "gt_span": [10, 20],
                    },
                }
            ]
        )
        assert metrics["overall"]["total"] == 1
        assert metrics["overall"]["mIoU"] == 100.0
        assert metrics["source=charades"]["total"] == 1

    def test_detect_unclosed_think(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("<think>unfinished") == ((-100.0, -100.0), True)

    def test_detect_stray_closing_think(self) -> None:
        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench))
        assert bench.parse_answer("12 - 34 </think>") == ((-100.0, -100.0), True)

    def test_dispatch_uses_override_when_registered(self) -> None:
        @ANSWER_PARSER_REGISTRY.register(name="external_timelens_bench")
        class ExternalTimeLensBench:
            def __call__(self, _: str, **__: object) -> Tuple[Tuple[float, float], bool]:
                return (((5.0, 6.0)), True)

        bench = TimeLensBench(Namespace(dataset_pp=DataSets.timelens_bench, answer_parser="external_timelens_bench"))
        assert bench.parse_answer("12 - 34") == (((5.0, 6.0)), True)
        ANSWER_PARSER_REGISTRY._obj_map.pop("external_timelens_bench", None)


class TestTVGBench:
    def test_build_dataset_from_timer1_json_format(self, tmp_path: Path, monkeypatch) -> None:
        anno_dir = tmp_path / "annotations"
        anno_dir.mkdir()
        video_root = tmp_path / "tvgbench_data"
        video_root.mkdir()
        (video_root / "KWZSA.mp4").write_bytes(b"")
        anno_path = anno_dir / "tvgbench.json"
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "path": "./dataset/timer1/videos/tvgbench_data/KWZSA.mp4",
                        "question": "a person was sitting at a table.",
                        "answer": "13.4-28.1",
                        "start": None,
                        "end": None,
                        "duration": 30.12,
                        "question_type": "video temporal grounding",
                        "source": "./dataset/timer1/videos/annotations/tvgbench/raw_data/Charades_sta_test.json",
                        "dataset_name": "charades",
                        "qsemtype": "Human Pose",
                        "bins": {"dataset": "charades", "qsemtype": "HP"},
                    }
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("TVGBENCH_VIDEO_ROOT", str(video_root))
        monkeypatch.setenv("TVGBENCH_INPUT_MODE", "video_path")
        bench = TVGBench(Namespace(dataset_dir=str(anno_path), answer_parser=""))
        rows = bench.build_dataset()

        assert len(rows) == 1
        assert rows[0]["videos"] == [str(video_root / "KWZSA.mp4")]
        assert rows[0]["pp_meta"]["gt_dataset"] == DataSets.tvgbench
        assert rows[0]["pp_meta"]["gt_source_dataset"] == "charades"
        assert rows[0]["pp_meta"]["gt_span"] == [13.4, 28.1]
        assert rows[0]["pp_meta"]["gt_question"] == "a person was sitting at a table."
        assert "Please find the visual event" in rows[0]["messages"][0]["content"]

    def test_uses_bins_dataset_when_dataset_name_missing(self, tmp_path: Path, monkeypatch) -> None:
        video_root = tmp_path / "videos" / "tvgbench_data"
        video_root.mkdir(parents=True)
        (video_root / "egonlq.mp4").write_bytes(b"")
        anno_path = tmp_path / "tvgbench.json"
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "path": "./dataset/timer1/videos/tvgbench_data/egonlq.mp4",
                        "question": "open the cabinet.",
                        "answer": "2-4",
                        "duration": 8.5,
                        "dataset_name": None,
                        "bins": {"dataset": "egonlq"},
                    }
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("TVGBENCH_VIDEO_ROOT", str(video_root))
        monkeypatch.setenv("TVGBENCH_INPUT_MODE", "video_path")
        rows = TVGBench(Namespace(dataset_dir=str(anno_path), answer_parser="")).build_dataset()

        assert len(rows) == 1
        assert rows[0]["pp_meta"]["gt_source_dataset"] == "egonlq"

    def test_calc_metric_reuses_timelens_metrics(self) -> None:
        bench = TVGBench(Namespace(dataset_pp=DataSets.tvgbench))
        metrics = bench.calc_metric(
            [
                {
                    "response": "<answer>From <t>10</t>s to <t>20</t>s</answer>",
                    "pp_meta": {
                        "gt_dataset": DataSets.tvgbench,
                        "gt_source_dataset": "charades",
                        "gt_span": [10, 20],
                    },
                }
            ]
        )
        assert metrics["overall"]["total"] == 1
        assert metrics["overall"]["mIoU"] == 100.0
        assert metrics["source=charades"]["R@0.5"] == 100.0

    def test_video_o3_multi_image_uses_tg_direct_prompt(self, monkeypatch) -> None:
        def fake_sampler(**_: object):
            return ["frame_000001.jpg"], {
                "sample_fps": 1.0,
                "sample_n_frames": 1,
                "duration": 5.0,
            }

        monkeypatch.setattr("openo3_eval.video_o3.message.sample_full_video_frames", fake_sampler)
        args = Namespace(
            model="openo3videov2_sft_135k",
            fps=1.0,
            frames_limit=768,
        )
        message, _ = build_video_o3_message(
            args,
            {"videos": ["fixtures/video.mp4"], "pp_meta": {"gt_question": "open the cabinet"}},
            "tvgbench",
        )

        assert message[0] == {"type": "question_type", "value": "tg"}
        assert message[2]["type"] == "direct_prompt"
        assert "locate the video segment" in message[2]["value"]
        assert "From <t>start_time</t>s to <t>end_time</t>s" in message[2]["value"]
        assert "Options:" not in message[-1]["value"]


class TestVideoMMEAnswerParsing:
    def test_parse_module_default_matches_dataset_default(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("A") == "A"

    def test_build_sample_json_file_dataset(self, tmp_path: Path, monkeypatch) -> None:
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        (video_dir / "demo.mp4").write_bytes(b"")
        anno_path = tmp_path / "Video-MME_sample270.json"
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "index": 1,
                        "video": "demo",
                        "video_path": "./video/demo.mp4",
                        "duration": "short",
                        "domain": "Knowledge",
                        "candidates": ["A. run", "B. sit"],
                        "sub_category": "Action",
                        "task_type": "Action Recognition",
                        "question": "What does the person do?",
                        "answer": "A",
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("VIDEO_MME_INPUT_MODE", "video_path")
        monkeypatch.setenv("VIDEO_MME_USER_PROMPT", "{question}\n{choice}\n{asr}")
        bench = VideoMME(Namespace(dataset_dir=str(anno_path), answer_parser=""))
        rows = bench.build_dataset()
        assert len(rows) == 1
        assert rows[0]["videos"] == [str(video_dir / "demo.mp4")]
        assert rows[0]["pp_meta"]["gt_question"] == "What does the person do?"
        assert rows[0]["pp_meta"]["gt_options_text"] == "A. run\nB. sit"

    def test_parse_parenthesized_letter(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("(B)") == "B"

    def test_parse_answer_tag(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("<answer>C</answer>") == "C"

    def test_parse_answer_prefix(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("Answer: D") == "D"

    def test_parse_candidate_text_fallback(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        candidates = ["red car", "blue bus", "green bike", "yellow train"]
        assert bench.parse_answer("blue bus", candidates=candidates) == "B"

    def test_parse_invalid_text_returns_empty_string(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("I am not sure") == ""

    def test_parse_extended_option_range(self) -> None:
        bench = VideoMME(Namespace(dataset_pp=DataSets.video_mme))
        assert bench.parse_answer("Final answer: H") == "H"


class TestVideoMMEv2AnswerParsing:
    def test_build_sample_json_file_dataset(self, tmp_path: Path, monkeypatch) -> None:
        video_dir = tmp_path / "videos"
        subtitle_dir = tmp_path / "subtitles" / "subtitle"
        video_dir.mkdir()
        subtitle_dir.mkdir(parents=True)
        (video_dir / "001.mp4").write_bytes(b"")
        (subtitle_dir / "001.jsonl").write_text('{"text":"hello","start_time":0,"end_time":1}\n', encoding="utf-8")
        anno_path = tmp_path / "Video-MME-v2_sample.json"
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "video_id": "001",
                        "question_id": "001-1",
                        "question": "What is shown?",
                        "options": "A. cat\nB. dog\nC. bird\nD. fish\nE. car\nF. bus\nG. train\nH. plane",
                        "answer": "H",
                        "group_type": "relevance",
                        "group_structure": "[1,2,3,4]",
                        "level": 1,
                        "second_head": "Perception",
                        "third_head": "Object",
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("VIDEO_MME_V2_INPUT_MODE", "video_path")
        bench = VideoMMEv2(Namespace(dataset_dir=str(anno_path), answer_parser=""))
        rows = bench.build_dataset()
        assert len(rows) == 1
        assert rows[0]["videos"] == [str(video_dir / "001.mp4")]
        assert rows[0]["pp_meta"]["gt_question_id"] == "001-1"
        assert rows[0]["pp_meta"]["gt_candidates"][-1] == "H. plane"

    def test_calc_metric_uses_group_rating(self) -> None:
        bench = VideoMMEv2(Namespace(dataset_pp=DataSets.video_mme_v2))
        rows = []
        answers = ["A", "B", "C", "D"]
        responses = ["A", "B", "wrong", "D"]
        for idx, (answer, response) in enumerate(zip(answers, responses, strict=True), start=1):
            rows.append(
                {
                    "response": response,
                    "pp_meta": {
                        "gt_answer": answer,
                        "gt_candidates": [
                            "A. one",
                            "B. two",
                            "C. three",
                            "D. four",
                            "E. five",
                            "F. six",
                            "G. seven",
                            "H. eight",
                        ],
                        "gt_question_id": f"001-{idx}",
                        "gt_video": "001",
                        "gt_group_type": "relevance",
                        "gt_group_structure": "[1,2,3,4]",
                        "gt_level": "1",
                        "gt_second_head": "Perception",
                        "gt_third_head": "Object",
                    },
                }
            )

        metrics = bench.calc_metric(rows)
        assert metrics["overall"]["acc"] == 75.0
        assert metrics["final_rating"]["total"] == 56.25
        assert metrics["level/1"]["score"] == 56.25


class TestVideoHolmesAnswerParsing:
    def test_parse_module_default_matches_dataset_default(self) -> None:
        bench = VideoHolmes(Namespace(dataset_pp=DataSets.video_holmes))
        assert bench.parse_answer("A") == "A"

    def test_build_sample_json_file_dataset(self, tmp_path: Path, monkeypatch) -> None:
        video_dir = tmp_path / "video"
        video_dir.mkdir()
        (video_dir / "demo.mp4").write_bytes(b"")
        anno_path = tmp_path / "test_Video-Holmos-sample180.json"
        anno_path.write_text(
            json.dumps(
                [
                    {
                        "video ID": "demo",
                        "Question ID": 1,
                        "Question Type": "SR",
                        "Question": "Who is Benjamin?",
                        "Options": {"A": "neighbor", "F": "friend"},
                        "Answer": "F",
                        "Explanation": "demo",
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("VIDEO_HOLMES_INPUT_MODE", "video_path")
        bench = VideoHolmes(Namespace(dataset_dir=str(anno_path), answer_parser=""))
        rows = bench.build_dataset()
        assert len(rows) == 1
        assert rows[0]["videos"] == [str(video_dir / "demo.mp4")]
        assert rows[0]["pp_meta"]["gt_options_text"] == "A. neighbor\nF. friend"

    def test_parse_answer_tag(self) -> None:
        bench = VideoHolmes(Namespace(dataset_pp=DataSets.video_holmes))
        assert bench.parse_answer("<answer>F</answer>") == "F"

    def test_parse_answer_prefix(self) -> None:
        bench = VideoHolmes(Namespace(dataset_pp=DataSets.video_holmes))
        assert bench.parse_answer("The answer is E.") == "E"

    def test_parse_candidate_text_fallback(self) -> None:
        bench = VideoHolmes(Namespace(dataset_pp=DataSets.video_holmes))
        candidates = ["A. red car", "B. blue bus", "C. green bike", "D. yellow train", "E. white boat", "F. black cat"]
        assert bench.parse_answer("white boat", candidates=candidates) == "E"


class TestCharadesTemporalGrounding:
    def test_parse_result_json(self) -> None:
        bench = Charades(Namespace(dataset_pp=DataSets.charades))
        assert bench.parse_answer('<answer>{"result": [[10, 20], [30.5, 40]]}</answer>') == [
            [10.0, 20.0],
            [30.5, 40.0],
        ]

    def test_parse_temporal_segment_and_scale_normalized_interval(self) -> None:
        bench = Charades(Namespace(dataset_pp=DataSets.charades))
        assert bench.parse_answer('{"temporal_segment": [0.25, 0.5]}', duration=80) == [[20.0, 40.0]]

    def test_calc_metric_matches_charades_iou_summary(self) -> None:
        bench = Charades(Namespace(dataset_pp=DataSets.charades))
        metrics = bench.calc_metric(
            [
                {
                    "response": '{"result": [[10, 20]]}',
                    "pp_meta": {"gt_intervals": [[10, 20]], "gt_duration": 100},
                },
                {
                    "response": "no valid segment",
                    "pp_meta": {"gt_intervals": [[30, 40]], "gt_duration": 100},
                },
            ]
        )
        assert metrics["overall"]["total"] == 2
        assert metrics["overall"]["valid"] == 1
        assert metrics["overall"]["miou"] == 1.0
        assert metrics["overall"]["R@1_IoU=0.5"] == 50.0

    def test_video_o3_message_uses_temporal_grounding_prompt(self, monkeypatch) -> None:
        def fake_sampler(**_: object):
            return ["frame_000001.jpg", "frame_000002.jpg"], {
                "sample_fps": 2.0,
                "sample_n_frames": 2,
                "duration": 5.0,
                "raw_fps": 4.0,
                "indices": [0, 8],
            }

        monkeypatch.setattr("openo3_eval.video_o3.message.sample_full_video_frames", fake_sampler)
        args = Namespace(model="openo3videov2_sft_135k", fps=2.0, frames_limit=768)
        message, _ = build_video_o3_message(
            args,
            {"videos": ["fixtures/video.mp4"], "pp_meta": {"gt_question": "open the door"}},
            "charades",
        )
        assert message[0] == {"type": "question_type", "value": "tg"}
        assert not any(item["type"] == "video" for item in message)
        assert message[2]["type"] == "direct_prompt"
        assert "Do not request more observations" in message[2]["value"]
        assert message[3]["type"] == "think_answer_prompt"
        assert "<think>" in message[3]["value"]
        assert message[4] == {"type": "text", "value": "<0.0 seconds><image>\n"}
        assert message[5]["type"] == "image"
        assert message[5]["value"] == "frame_000001.jpg"
        assert message[6] == {"type": "text", "value": "<2.0 seconds><image>\n"}
        assert message[7]["type"] == "image"
        assert message[7]["value"] == "frame_000002.jpg"
        assert "locate the video segment according to the given query" in message[-1]["value"]
        assert "Options:" not in message[-1]["value"]

    def test_video_o3_frame_timestamps_fall_back_to_sample_fps_rule(self) -> None:
        timestamps = frame_timestamps_from_meta({"sample_fps": 2.0, "duration": 0.25, "indices": [0, 8]}, 2)
        assert timestamps == [0.0, 0.25]

    def test_video_o3_multi_image_message_includes_direct_prompt(self, monkeypatch) -> None:
        def fake_sampler(**_: object):
            return ["frame_000001.jpg", "frame_000002.jpg"], {
                "sample_fps": 1.0,
                "sample_n_frames": 2,
                "duration": 5.0,
                "raw_fps": 2.0,
                "indices": [0, 2],
            }

        monkeypatch.setattr("openo3_eval.video_o3.message.sample_full_video_frames", fake_sampler)
        args = Namespace(
            model="openo3videov2_sft_135k",
            fps=1.0,
            frames_limit=768,
        )
        message, _ = build_video_o3_message(
            args,
            {
                "videos": ["fixtures/video.mp4"],
                "pp_meta": {
                    "gt_question": "What happens next?",
                    "gt_options_text": "A. run\nB. sit",
                },
            },
            "video_holmes",
        )
        assert message[0] == {"type": "raw_video", "value": "fixtures/video.mp4"}
        assert message[1]["type"] == "direct_prompt"
        assert "Do not request more observations" in message[1]["value"]
        assert message[2]["type"] == "think_answer_prompt"
        assert "<think>" in message[2]["value"] and "<answer>" in message[2]["value"]
        assert message[3] == {"type": "text", "value": "<0.0 seconds><image>\n"}
        assert not any(item["type"] == "video" for item in message)
        assert message[4]["type"] == "image"
        assert message[4]["min_pixels"] == 20 * 32 * 32
        assert message[4]["max_pixels"] == 768 * 32 * 32
        assert message[4]["total_pixels"] == 16384 * 32 * 32
        assert "Here are uniformly sampled frames" in message[-1]["value"]
        assert "What happens next?" in message[-1]["value"]

    def test_video_o3_multi_image_temporal_message_includes_tg_direct_prompt(self, monkeypatch) -> None:
        def fake_sampler(**_: object):
            return ["frame_000001.jpg"], {
                "sample_fps": 1.0,
                "sample_n_frames": 1,
                "duration": 5.0,
            }

        monkeypatch.setattr("openo3_eval.video_o3.message.sample_full_video_frames", fake_sampler)
        args = Namespace(
            model="openo3videov2_sft_135k",
            fps=1.0,
            frames_limit=768,
        )
        message, _ = build_video_o3_message(
            args,
            {"videos": ["fixtures/video.mp4"], "pp_meta": {"gt_question": "open the door"}},
            "timelens_bench",
        )
        assert message[0] == {"type": "question_type", "value": "tg"}
        assert message[2]["type"] == "direct_prompt"
        assert "locate the video segment" in message[2]["value"]
        assert "From <t>start_time</t>s to <t>end_time</t>s" in message[2]["value"]
        assert message[3]["type"] == "think_answer_prompt"
        assert "<think>" in message[3]["value"]
        assert "From <t>start_time</t>s to <t>end_time</t>s" in message[3]["value"]

    def test_video_o3_multi_image_config_matches_reference_defaults(self) -> None:
        config = VIDEO_O3_MODEL_CONFIGS["openo3videov2_sft_135k"]
        assert config.default_max_turn == 5
        assert config.total_pixels == 16384 * 32 * 32
        assert config.system_prompt is not None and "visual grounding" in config.system_prompt

    def test_video_o3_qwen35_config_uses_custom_tool_format(self) -> None:
        config = VIDEO_O3_MODEL_CONFIGS["qwen3.5-35b-a3b-sft-62k"]
        assert config.default_max_turn == 0
        assert config.tool_format == "qwen3_5"
        assert config.runtime_kwargs == {"max_num_seqs": 1}
        assert config.system_prompt is not None and "visual grounding" in config.system_prompt

    def test_video_o3_multi_image_tg_uses_configured_tg_prompt(self) -> None:
        config = VIDEO_O3_MODEL_CONFIGS["openo3videov2_sft_135k"]
        model = object.__new__(VideoO3MultiImageChat)
        model.system_prompt = config.system_prompt
        model.tg_system_prompt = config.tg_system_prompt
        prompt = model._system_prompt_for_question_type("tg")
        assert prompt is not None
        assert "From <t>start_time</t>s to <t>end_time</t>s" in prompt

    def test_video_o3_multi_image_processor_uses_dynamic_video_pixel_budget(self, monkeypatch) -> None:
        config = VIDEO_O3_MODEL_CONFIGS["openo3videov2_sft_135k"]
        model = object.__new__(VideoO3MultiImageChat)
        model.min_pixels = config.min_pixels
        model.max_pixels = config.max_pixels
        model.total_pixels = config.total_pixels
        model.use_dynamic_quota = config.use_dynamic_quota
        monkeypatch.setattr("openo3_eval.video_o3.model_multi_image.load_local_or_url_image", lambda image: image)
        image_items = [
            {
                "type": "image",
                "image": f"{idx}.jpg",
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
                "total_pixels": config.total_pixels,
            }
            for idx in range(768)
        ]
        assert model._image_processor_pixel_kwargs(image_items) == {"min_pixels": 20480, "max_pixels": 43690}

        messages = [{"role": "user", "content": image_items[:2]}]
        images, processor_kwargs = model._collect_images_and_processor_kwargs(messages)
        assert images == ["0.jpg", "1.jpg"]
        assert processor_kwargs == {"min_pixels": 20480, "max_pixels": 786432}

    def test_video_o3_qwen35_large_model_config_is_isolated(self) -> None:
        config = VIDEO_O3_MODEL_CONFIGS["qwen3.5-35b-a3b"]
        kwargs = config.to_kwargs()
        assert config.default_max_turn == 0
        assert kwargs["tool_format"] == "qwen3_5"
        assert kwargs["max_num_seqs"] == 1
        assert kwargs["fallback_mode"] == "think_then_answer"

        reference_kwargs = VIDEO_O3_MODEL_CONFIGS["qwen3_5-cot"].to_kwargs()
        assert "tool_format" not in reference_kwargs
        assert "max_num_seqs" not in reference_kwargs
        assert VIDEO_O3_MODEL_CONFIGS["qwen3.5-35b-a3b"].to_kwargs() == kwargs

    def test_video_o3_multi_image_extract_tool_call_info_supports_qwen35_format(self) -> None:
        model = object.__new__(VideoO3MultiImageChat)
        response = (
            "<think>need closer look</think>\n"
            "<tool_call>\n"
            "<function=crop_video>\n"
            "<parameter=start_time>\n12\n</parameter>\n"
            "<parameter=end_time>\n18\n</parameter>\n"
            '<parameter=sampling_strategy>\n"fine"\n</parameter>\n'
            "</function>\n"
            "</tool_call>\n"
        )

        assert model.extract_tool_call_info(response) == (12.0, 18.0, "fine")


class TestVideoO3ResponseMetadata:
    def test_strip_video_o3_metadata(self) -> None:
        response, metadata = strip_video_o3_metadata(
            "C\n[TURN_COUNT: 3]\n[INFERENCE_TIME: 12.50s]\n[INFERENCE_TIME_GENERATE: 9.25s]"
        )
        assert response == "C"
        assert metadata == {
            "dialogue_turns": [],
            "turn_count": 3,
            "inference_time": 12.5,
            "inference_time_generate": 9.25,
        }
