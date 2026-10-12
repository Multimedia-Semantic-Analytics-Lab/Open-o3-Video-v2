import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import convert_stgr, convert_trajectory, resolve_media, sample_frames  # noqa: E402


class PrepareDataTests(unittest.TestCase):
    def test_media_manifest_and_stgr_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()
            manifest = {"source/clip.mp4": str(video)}
            self.assertEqual(resolve_media("source/clip.mp4", root, manifest), video)
            row = {
                "id": "sample", "video_path": "source/clip.mp4", "question": "When?",
                "reasoning_process": "The person appears at the start.", "answer": [1, 4],
                "task": "temporal QA",
            }
            result = convert_stgr(row, root, manifest, root / "cache", 96, True)
            self.assertEqual(result["messages"][1]["content"], "When?")
            self.assertIn("<answer>From <t>1</t>s to <t>4</t>s</answer>", result["messages"][2]["content"])

    def test_key_frames_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            (root / "key.jpg").touch()
            row = {
                "id": "sample", "task": "temporal-spatial free-form QA",
                "video_path": "clip.mp4", "question": "What happens?", "answer": "A person enters.",
                "key_frames": [{"time": 2.5, "path": "key.jpg"}],
            }
            convert_stgr(row, root, {}, root / "cache", 96, True)
            (root / "key.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                convert_stgr(row, root, {}, root / "cache", 96, True)

    def test_trajectory_requires_one_video_per_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            row = {
                "id": "sample", "messages": [{"role": "user", "content": "<video>"}],
                "videos": [{"url": "clip.mp4", "crop": [-1, -1], "sample": "full"}],
            }
            result = convert_trajectory(row, root, {}, root / "cache", 96, True)
            self.assertEqual(result["images"], [])
            self.assertEqual(result["messages"][0]["content"], "[video validated]")
            row["videos"] = []
            with self.assertRaisesRegex(ValueError, "markers"):
                convert_trajectory(row, root, {}, root / "cache", 96, True)

    def test_trajectory_keeps_frames_from_each_turn_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "initial.mp4").touch()
            (root / "crop.mp4").touch()
            initial_frames = [str(root / "initial-1.jpg"), str(root / "initial-2.jpg")]
            crop_frames = [str(root / "crop-1.jpg")]
            row = {
                "messages": [
                    {"role": "user", "content": "Inspect <video>"},
                    {"role": "assistant", "content": "I need a closer look."},
                    {"role": "tool", "content": "Crop result: <video>"},
                    {"role": "assistant", "content": "The answer is visible."},
                ],
                "videos": [
                    {"url": "initial.mp4", "crop": [-1, -1], "sample": "full"},
                    {"url": "crop.mp4", "crop": [10, 11], "sample": "fine"},
                ],
            }
            with patch("prepare_data.sample_frames", side_effect=[initial_frames, crop_frames]):
                result = convert_trajectory(row, root, {}, root / "cache", 96, False)
            contents = [message["content"] for message in result["messages"]]
            self.assertEqual([content.count("<image>") for content in contents], [2, 0, 1, 0])
            self.assertEqual(result["images"], initial_frames + crop_frames)
            self.assertIn("<t>10</t>s: <image>", contents[2])

    def test_frame_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()

            def fake_run(command, check):
                self.assertTrue(check)
                self.assertIn("-frames:v", command)
                Path(command[-1].replace("%04d", "0001")).touch()

            with patch("prepare_data.shutil.which", return_value="ffmpeg"), patch(
                "prepare_data.subprocess.run", side_effect=fake_run
            ) as run:
                first = sample_frames(video, [1, 3], "medium", root / "cache", 96)
                second = sample_frames(video, [1, 3], "medium", root / "cache", 96)
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)

    def test_crop_and_keyframe_timestamps_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            key = root / "key.jpg"
            key.touch()
            frames = [str(root / "0001.jpg"), str(root / "0002.jpg")]
            row = {
                "id": "sample", "task": "temporal-spatial free-form QA",
                "video_path": "clip.mp4", "question": "What happens?", "answer": "A person enters.",
                "key_frames": [{"time": 0.5, "path": "key.jpg"}],
            }
            with patch("prepare_data.sample_frames", return_value=frames):
                result = convert_stgr(row, root, {}, root / "cache", 96, False)
            self.assertEqual(result["images"], [frames[0], str(key), frames[1]])
            self.assertIn("<t>0.5</t>s: <image>", result["messages"][1]["content"])

            trajectory = {
                "messages": [{"role": "user", "content": "Inspect <video> and answer."}],
                "videos": [{"url": "clip.mp4", "crop": [10, 11], "sample": "fine"}],
            }
            with patch("prepare_data.sample_frames", return_value=frames):
                result = convert_trajectory(trajectory, root, {}, root / "cache", 96, False)
            self.assertEqual(result["images"], frames)
            self.assertIn("<t>10.25</t>s: <image>", result["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
