from argparse import Namespace

from openo3_eval.datasetpp.base import BaseDatasetPP


def test_build_user_prompt_replaces_video_placeholder() -> None:
    bench = BaseDatasetPP(Namespace(dataset_dir=""))
    prompt = bench.build_user_prompt(
        default="<video>\n<ASR>{asr_info}</ASR>",
        prompt_replacements={"<video>": "<0.0 seconds><image>"},
        asr_info="a",
    )
    assert "<0.0 seconds><image>" in prompt
    assert "<video>" not in prompt
