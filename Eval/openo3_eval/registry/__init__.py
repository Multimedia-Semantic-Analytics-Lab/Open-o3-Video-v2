from enum import Enum

from openo3_eval.registry.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
ANSWER_PARSER_REGISTRY = Registry("ANSWER_PARSER")


class AnswerParsers(str, Enum):
    timelens_bench_parser = "timelens_bench_parser"
    video_mme_parser = "video_mme_parser"
    video_holmes_parser = "video_holmes_parser"
    charades_parser = "charades_parser"
    videommmu_parser = "videommmu_parser"


class DataSets(str, Enum):
    timelens_bench = "timelens_bench"
    video_mme = "video_mme"
    video_mme_v2 = "video_mme_v2"
    lvbench = "lvbench"
    video_holmes = "video_holmes"
    charades = "charades"
    tvgbench = "tvgbench"
    videozerobench = "videozerobench"
    v_star = "v_star"
    worldsense = "worldsense"
    videommmu = "videommmu"
    longvideobench = "longvideobench"
