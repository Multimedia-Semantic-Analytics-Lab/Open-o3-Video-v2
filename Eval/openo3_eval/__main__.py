import importlib.util
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

ROUTE_MAPPING: Dict[str, str] = {
    "eval-openo3": "openo3_eval.cli.eval_openo3",
    "view": "openo3_eval.cli.view",
}


def use_torchrun() -> bool:
    nproc_per_node = os.getenv("NPROC_PER_NODE")
    nnodes = os.getenv("NNODES")
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return None
    torchrun_args: List[str] = []
    for env_key in ["NPROC_PER_NODE", "MASTER_PORT", "NNODES", "NODE_RANK", "MASTER_ADDR"]:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f"--{env_key.lower()}", env_val]
    return torchrun_args


def prepare_config_args(argv: List[str]) -> None:
    for i in range(len(argv)):
        if argv[i] == "--config":
            if i + 1 >= len(argv):
                raise ValueError("The `--config` argument requires a yaml file path.")
            from omegaconf import DictConfig, ListConfig, OmegaConf

            config = OmegaConf.load(argv[i + 1])

            def parse_dict_config(cfg: DictConfig) -> Dict[str, Any]:
                result: Dict[str, Any] = {}
                for key, value in cfg.items():
                    if isinstance(value, DictConfig):
                        result[str(key)] = json.dumps(OmegaConf.to_container(value))
                    elif isinstance(value, ListConfig):
                        result[str(key)] = list(value)
                    else:
                        result[str(key)] = value
                return result

            # Convert yaml to cmd line
            assert isinstance(config, DictConfig), "config must be a DictConfig"
            cfg = parse_dict_config(config)
            for key, value in cfg.items():
                argv.append(f"--{key}")
                if isinstance(value, list):
                    argv.extend(value)
                else:
                    argv.append(str(value))

            # Pop --config
            argv.pop(i)
            # Pop value of --config
            argv.pop(i)
            break


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    if not argv:
        supported = ", ".join(sorted(route_mapping))
        raise ValueError(f"[Open-o3 Eval] Missing command. Supported commands: {supported}")
    method_name = argv[0].replace("_", "-")
    argv = argv[1:]
    if method_name not in route_mapping:
        supported = ", ".join(sorted(route_mapping))
        raise ValueError(f"[Open-o3 Eval] Unknown command {method_name!r}. Supported commands: {supported}")
    module_name = route_mapping[method_name]
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Module {module_name} not found"
    file_path = spec.origin
    assert file_path is not None, f"Module {module_name} has no origin"
    torchrun_args = get_torchrun_args()
    prepare_config_args(argv)
    python_cmd = sys.executable
    args: List[str]
    if torchrun_args is None or method_name not in {"eval-openo3"}:
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, "-m", "torch.distributed.run", *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    try:
        cli_main()
    except ValueError as err:
        print(str(err), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
