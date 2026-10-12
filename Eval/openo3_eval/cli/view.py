import argparse
import json
import os
import sys
from typing import List

import openo3_eval.datasetpp
import openo3_eval.parser  # noqa: F401
from openo3_eval.datasetpp import BaseDatasetPP
from openo3_eval.registry import DATASET_REGISTRY
from openo3_eval.runtime import get_logger, import_external_file

logger = get_logger()


def import_external_plugins(external_plugins: List[str] | str) -> None:
    if isinstance(external_plugins, str):
        external_plugins = [external_plugins]
    if not external_plugins:
        return
    for external_plugin in external_plugins:
        import_external_file(external_plugin)
    logger.info(f"[Open-o3 Eval] Successfully imported external_plugins: {external_plugins}.")


def view_main() -> None:
    parser = argparse.ArgumentParser(description="[Open-o3 Eval] View and re-evaluate inference results")
    parser.add_argument("--result_path", required=True, help="Path to result jsonl file")
    parser.add_argument("--dataset_pp", required=True, help="Dataset plugin name")
    parser.add_argument(
        "--external_plugins",
        action="append",
        default=[],
        help="Path to an external plugin file; repeat to load multiple.",
    )
    parser.add_argument("--answer_parser", default="", help="Registered answer parser name to use.")
    args = parser.parse_args()

    import_external_plugins(args.external_plugins)

    if not os.path.exists(args.result_path):
        logger.error(f"[Open-o3 Eval] Error: result file not found: {args.result_path}")
        sys.exit(1)

    data_list = []
    with open(args.result_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))

    logger.info(f"[Open-o3 Eval] Loaded {len(data_list)} results from: {args.result_path}")

    ns = argparse.Namespace(dataset_pp=args.dataset_pp, answer_parser=args.answer_parser)
    dataset_pp: BaseDatasetPP = DATASET_REGISTRY.get(args.dataset_pp)(ns)

    metrics = dataset_pp.calc_metric(data_list)
    dataset_pp.print_metric(metrics, args.dataset_pp)


if __name__ == "__main__":
    view_main()
