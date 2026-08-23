"""统一入口：根据首个参数选择转换、在线推理或 LIBERO 评估。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot_inference import __version__


_COMMANDS = ("convert", "infer", "infer-once", "eval-libero")

_PROG_ALIASES = {
    "lerobot-convert": "convert",
    "lerobot_convert": "convert",
    "lerobot-infer-once": "infer-once",
    "lerobot_infer_once": "infer-once",
    "lerobot-infer": "infer",
    "lerobot_infer": "infer",
}


class _ChineseHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """保留多行说明；命令简述使用中文。"""


def _prog_name() -> str:
    return Path(sys.argv[0]).name


def _alias_command() -> str | None:
    name = _prog_name()
    if name in _PROG_ALIASES:
        return _PROG_ALIASES[name]
    # PyInstaller onedir 可执行名常见为 lerobot_infer
    stem = Path(name).stem
    return _PROG_ALIASES.get(stem)


def _localize_parser(parser: argparse.ArgumentParser) -> None:
    """将 argparse 默认英文分组标题与 -h 说明改为中文。"""
    parser._optionals.title = "可选参数"
    if parser._positionals.title in (None, "positional arguments"):
        parser._positionals.title = "位置参数"
    # 覆盖默认英文 help 动作说明
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            action.help = "显示帮助信息并退出"


def _add_subparser(subparsers: argparse._SubParsersAction, *args, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("formatter_class", _ChineseHelpFormatter)
    kwargs.setdefault("add_help", True)
    sp = subparsers.add_parser(*args, **kwargs)
    _localize_parser(sp)
    return sp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot",
        description=(
            "LeRobot 推理与转换统一工具。\n"
            "通过第一个参数选择功能：convert（转换）、infer（Dora 在线推理）、"
            "infer-once（本地单步 smoke）、eval-libero（LIBERO 闭环评估）。"
        ),
        epilog=(
            "示例:\n"
            "  lerobot convert --config examples/dora_convert/convert.yaml\n"
            "  lerobot infer --config examples/dora_infer_act_7d/policy_act.yaml\n"
            "  lerobot infer-once --config examples/dora_infer_act_7d/policy_act.yaml\n"
            "  lerobot eval-libero --config examples/libero_eval/policy_vla_jepa.yaml --libero-root /path/to/LIBERO\n"
            "\n"
            "兼容说明:\n"
            "  若通过 lerobot-convert / lerobot-infer / lerobot-infer-once 调用，\n"
            "  可省略子命令（按可执行文件名自动选择功能）。\n"
            "  二进制名含 lerobot_infer 且未写子命令时，默认按 infer 处理。"
        ),
        formatter_class=_ChineseHelpFormatter,
    )
    _localize_parser(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="显示版本信息并退出",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="功能",
        description="必须指定下列之一作为第一个参数",
        metavar="{convert,infer,infer-once,eval-libero}",
        required=True,
    )

    # —— convert ——
    p_convert = _add_subparser(
        subparsers,
        "convert",
        help="将 policy_train ACT 产物转为 LeRobot pretrained 目录",
        description=(
            "转换：读取 policy_train 的 *_standard.safetensors 与 dataset_stats，"
            "写出可直接用于推理的 pretrained 资产（model.safetensors / config / processor）。"
        ),
    )
    p_convert.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（其中相对路径相对该 yaml 所在目录解析）",
    )
    p_convert.add_argument(
        "--src-dir",
        type=str,
        default=None,
        help="policy_train 检查点目录（含 *_standard.safetensors）",
    )
    p_convert.add_argument(
        "--dst-dir",
        type=str,
        default=None,
        help=(
            "输出根目录；实际写入 "
            "{dst}/{prefix}/epoch_XXXXXX/（能解析轮次时，即使只有一个）"
            "或 {dst}/{prefix}/single/（无法解析轮次）；"
            "prefix 默认是 src_dir 目录名"
        ),
    )
    p_convert.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="输出子目录前缀（默认使用 src_dir 的目录名）",
    )
    p_convert.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "指定某个 .safetensors 文件名或路径（跳过自动发现）；"
            "能解析轮次则写入 {prefix}/epoch_XXXXXX/，否则 {prefix}/single/"
        ),
    )
    p_convert.add_argument(
        "--task-json",
        type=str,
        default=None,
        help=(
            "可选 policy_train task.json；省略时使用内置默认："
            "相机 observation.images.{left,right,top} + pick_and_place ACT 超参"
        ),
    )
    p_convert.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印映射报告，不写文件",
    )
    p_convert.set_defaults(_handler="convert")

    # —— infer ——
    p_infer = _add_subparser(
        subparsers,
        "infer",
        help="以 Dora 策略节点运行在线推理",
        description=(
            "推理：作为 Dora 节点订阅 proprio_state 与多路图像，"
            "加载 LeRobot pretrained_model/，输出 JointCommand。"
        ),
    )
    p_infer.add_argument(
        "--config",
        type=str,
        default=None,
        help="推理 YAML 配置路径（也可用环境变量 LEROOT_INFERENCE_CONFIG）",
    )
    p_infer.set_defaults(_handler="infer")

    # —— infer-once ——
    p_once = _add_subparser(
        subparsers,
        "infer-once",
        help="本地单步推理 smoke（不依赖 Dora）",
        description=(
            "单步 smoke：按配置加载策略，用随机观测跑一步 generate_action，"
            "打印 action JSON，用于验证环境与权重。"
        ),
    )
    p_once.add_argument(
        "--config",
        type=str,
        required=True,
        help="推理 YAML 配置文件路径",
    )
    p_once.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="覆盖 observation.state 长度（默认：关节数）",
    )
    p_once.add_argument(
        "--height",
        type=int,
        default=480,
        help="假图像高度（像素，默认 480）",
    )
    p_once.add_argument(
        "--width",
        type=int,
        default=640,
        help="假图像宽度（像素，默认 640）",
    )
    p_once.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（默认 0）",
    )
    p_once.add_argument(
        "--async-timeout",
        type=float,
        default=120.0,
        help="异步 backend 等待首个 action 的超时秒数（默认 120）",
    )
    p_once.set_defaults(_handler="infer-once")

    # —— eval-libero ——
    p_eval = _add_subparser(
        subparsers,
        "eval-libero",
        help="在本地 LIBERO 场景中闭环评估通用策略",
        description="LIBERO 评估：加载已注册 policy adapter，输出结果、动作日志和双相机视频。",
    )
    p_eval.add_argument("--config", type=str, required=True, help="通用策略推理 YAML 配置路径")
    p_eval.add_argument("--libero-root", type=str, required=True, help="本地 LIBERO 仓库根目录")
    p_eval.add_argument("--suite", type=str, default="libero_spatial", help="任务套件名称")
    p_eval.add_argument("--task-id", type=int, default=0, help="任务编号")
    p_eval.add_argument("--episodes", type=int, default=1, help="episode 数量")
    p_eval.add_argument("--init-state-id", type=int, default=0, help="首个初始状态编号")
    p_eval.add_argument("--seed", type=int, default=0, help="环境随机种子")
    p_eval.add_argument("--episode-length", type=int, default=280, help="每个 episode 最大步数；suite 专用 horizon 请通过协议 YAML 指定")
    p_eval.add_argument("--num-steps-wait", type=int, default=10, help="reset 后稳定步数")
    p_eval.add_argument("--control-freq", type=int, default=20, help="控制频率和视频帧率")
    p_eval.add_argument("--observation-size", type=int, default=256, help="相机渲染边长")
    p_eval.add_argument("--output-dir", type=str, default="examples/libero_eval/out", help="运行输出父目录")
    p_eval.set_defaults(_handler="eval-libero")

    return parser


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """保证 argv[0] 为子命令；兼容旧入口名自动注入。

    不注入的情况：
    - 已显式写出 convert / infer / infer-once / eval-libero
    - 仅要顶层元信息（无参数、-h / --help 或 --version）
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _COMMANDS:
        return args
    # 顶层帮助/版本：不要因可执行名 lerobot_infer 被误注入子命令。
    if not args or args[0] in ("-h", "--help", "--version"):
        return args
    alias = _alias_command()
    if alias is not None:
        return [alias, *args]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    handler = getattr(args, "_handler", None)

    if handler == "convert":
        from lerobot_inference.convert.cli import run_convert

        return run_convert(args)
    if handler == "infer":
        from lerobot_inference.inference.main import run_infer

        return run_infer(args)
    if handler == "infer-once":
        from lerobot_inference.inference.run_once import run_infer_once

        return run_infer_once(args)
    if handler == "eval-libero":
        from lerobot_inference.inference.libero_eval import run_libero_eval

        return run_libero_eval(args)

    parser.error("未知功能，请指定 convert / infer / infer-once / eval-libero")
    return 2


def main_convert(argv: list[str] | None = None) -> int:
    """兼容 console_script：lerobot-convert。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "convert":
        args = ["convert", *args]
    return main(args)


def main_infer(argv: list[str] | None = None) -> int:
    """兼容 console_script：lerobot-infer。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "infer":
        args = ["infer", *args]
    return main(args)


def main_infer_once(argv: list[str] | None = None) -> int:
    """兼容 console_script：lerobot-infer-once。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "infer-once":
        args = ["infer-once", *args]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(main())
