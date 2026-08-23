"""通用 LIBERO 闭环评估器。"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, TextIO

import numpy as np
import yaml
from PIL import Image, ImageDraw

from lerobot_inference.inference.config import load_config
from lerobot_inference.inference.policies.registry import (
    create_policy_adapter,
    normalize_policy_type,
)

logger = logging.getLogger("lerobot_inference.libero_eval")

_CAMERA_KEYS = ("image", "image2")
_STATE_DIM = 8
_ACTION_DIM = 7


class LiberoEpisodeEnv(Protocol):
    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def close(self) -> None: ...


class EvaluationPolicy(Protocol):
    def reset(self) -> None: ...


    def is_observation_needed(self) -> bool: ...

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray: ...


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _make_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _configure_logging(run_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def prepare_libero_runtime(libero_root: str | Path, run_dir: Path) -> dict[str, str]:
    """将 LIBERO 全局路径配置绑定到指定本地 checkout。"""
    root = Path(libero_root).expanduser().resolve()
    package_root = root / "libero"
    benchmark_root = package_root / "libero"
    required = {
        "LIBERO package root": package_root,
        "benchmark root": benchmark_root / "__init__.py",
        "BDDL files": benchmark_root / "bddl_files",
        "initial states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Invalid LIBERO root; missing " + "; ".join(missing))

    config_dir = run_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = config_dir / "datasets"
    matplotlib_dir = config_dir / "matplotlib"
    datasets_dir.mkdir(exist_ok=True)
    matplotlib_dir.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(benchmark_root / "assets"),
    }
    _write_yaml_atomic(config_dir / "config.yaml", paths)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    root_text = str(root)
    sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != root_text]
    sys.path.insert(0, root_text)
    _avoid_robosuite_global_log_collision()
    return {"libero_root": root_text, "config_dir": str(config_dir), **paths}


def _avoid_robosuite_global_log_collision() -> None:
    """使用 robosuite 的 private-macros hook，避免修改 site-packages。

    Robosuite 1.4 在缺少 ``macros_private.py`` 时会写入全局日志文件。
    这里仅注入空模块以关闭回退日志，不修改已安装的实际私有配置。
    """
    if "robosuite" in sys.modules or "robosuite.macros_private" in sys.modules:
        return
    spec = importlib.util.find_spec("robosuite")
    search_locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if any((Path(location) / "macros_private.py").is_file() for location in search_locations):
        return
    private_macros = ModuleType("robosuite.macros_private")
    private_macros.__file__ = "<lerobot-inference runtime shim>"
    sys.modules["robosuite.macros_private"] = private_macros


def validate_libero_module_origin(module: ModuleType, libero_root: str | Path) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Imported libero.libero has no __file__; cannot validate its source")
    origin = Path(module_file).resolve()
    expected = (Path(libero_root).expanduser().resolve() / "libero").resolve()
    if not origin.is_relative_to(expected):
        raise RuntimeError(
            "LIBERO import did not come from --libero-root: "
            f"imported={origin}, expected_under={expected}"
        )
    return origin


def import_local_libero(libero_root: str | Path) -> ModuleType:
    already_loaded = sys.modules.get("libero.libero")
    if already_loaded is not None:
        validate_libero_module_origin(already_loaded, libero_root)
        return already_loaded
    module = importlib.import_module("libero.libero")
    validate_libero_module_origin(module, libero_root)
    return module


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"Expected one finite quaternion in xyzw order, got shape={quat.shape}")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(w) / denominator)).astype(np.float32)


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB LIBERO image, got shape={array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 LIBERO image, got dtype={array.dtype}")
    return np.ascontiguousarray(array[::-1, ::-1])


def build_libero_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """将 LIBERO 原始观测转换为 adapter 共用的 LeRobot 特征。"""
    pixels = observation["pixels"]
    robot_state = observation["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    eef_quat = np.asarray(robot_state["eef"]["quat"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    if eef_pos.shape != (3,) or gripper_qpos.shape != (2,):
        raise ValueError(
            "LIBERO-derived state must be a finite 8D vector; invalid shapes: "
            f"eef_pos={eef_pos.shape}, gripper_qpos={gripper_qpos.shape}"
        )
    state = np.concatenate((eef_pos, quat_to_axis_angle(eef_quat), gripper_qpos)).astype(
        np.float32, copy=False
    )
    if state.shape != (_STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("LIBERO-derived state must be a finite 8D vector")
    result: dict[str, np.ndarray] = {"observation.state": state}
    for key in _CAMERA_KEYS:
        result[f"observation.images.{key}"] = rotate_libero_image(pixels[key])
    return result


def make_rollout_frame(
    observation: dict[str, Any],
    *,
    task_description: str,
    step: int,
    replans: int,
    success: bool,
) -> np.ndarray:
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise KeyError("LIBERO observation is missing pixels")
    images = [rotate_libero_image(pixels[key]) for key in _CAMERA_KEYS]
    labeled: list[np.ndarray] = []
    for camera_name, image in zip(("agentview", "wrist"), images, strict=True):
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle((0, 0, pil_image.width, 46), fill=(0, 0, 0))
        draw.text((8, 5), camera_name, fill=(255, 255, 255))
        draw.text(
            (8, 23),
            f"step={step} replans={replans} success={str(success).lower()}",
            fill=(255, 255, 255),
        )
        labeled.append(np.asarray(pil_image))
    frame = np.concatenate(labeled, axis=1)
    pil_frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_frame)
    task_text = task_description[:100]
    text_width = min(pil_frame.width, max(1, 8 + len(task_text) * 7))
    draw.rectangle((0, pil_frame.height - 22, text_width, pil_frame.height), fill=(0, 0, 0))
    draw.text((8, pil_frame.height - 18), task_text, fill=(255, 255, 255))
    return np.asarray(pil_frame)


class RolloutVideoWriter:
    def __init__(self, path: Path, fps: int) -> None:
        import imageio.v2 as imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = imageio.get_writer(str(path), fps=fps)

    def append(self, frame: np.ndarray) -> None:
        self._writer.append_data(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> RolloutVideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


VideoWriterFactory = Callable[[Path, int], RolloutVideoWriter]


def _write_action_row(action_log: TextIO, payload: dict[str, Any]) -> None:
    action_log.write(json.dumps(payload, ensure_ascii=False) + "\n")
    action_log.flush()


def run_single_episode(
    *,
    env: LiberoEpisodeEnv,
    policy: EvaluationPolicy,
    task_description: str,
    episode_index: int,
    init_state_id: int,
    seed: int,
    episode_length: int,
    control_freq: int,
    video_path: Path,
    action_log: TextIO,
    video_writer_factory: VideoWriterFactory = RolloutVideoWriter,
) -> dict[str, Any]:
    policy.reset()
    observation, reset_info = env.reset(seed=seed)
    del reset_info
    replans = 0
    inference_latencies: list[float] = []
    action_latencies: list[float] = []
    action_min = np.full(_ACTION_DIM, np.inf, dtype=np.float64)
    action_max = np.full(_ACTION_DIM, -np.inf, dtype=np.float64)
    success = False
    termination = "horizon"
    steps = 0
    started = time.monotonic()

    with video_writer_factory(video_path, control_freq) as video:
        video.append(
            make_rollout_frame(
                observation,
                task_description=task_description,
                step=0,
                replans=0,
                success=False,
            )
        )
        for step_index in range(episode_length):
            fresh_prediction = bool(policy.is_observation_needed())
            policy_observation = build_libero_observation(observation) if fresh_prediction else {}
            action_started = time.monotonic()
            action = np.asarray(
                policy.generate_action(policy_observation, list(_CAMERA_KEYS)),
                dtype=np.float32,
            )
            action_latency = time.monotonic() - action_started
            action_latencies.append(action_latency)
            if fresh_prediction:
                replans += 1
                inference_latencies.append(action_latency)

            if action.shape != (_ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError(
                    "Policy must return one finite 7D LIBERO action: "
                    f"shape={action.shape}, finite={bool(np.isfinite(action).all())}"
                )
            action_min = np.minimum(action_min, action)
            action_max = np.maximum(action_max, action)
            observation, reward, terminated, truncated, info = env.step(action)
            steps = step_index + 1
            success = bool(info.get("is_success", False))
            _write_action_row(
                action_log,
                {
                    "episode": episode_index,
                    "init_state_id": init_state_id,
                    "step": step_index,
                    "new_prediction": fresh_prediction,
                    "action_latency_s": action_latency,
                    "action": action.tolist(),
                    "action_min": float(action.min()),
                    "action_max": float(action.max()),
                    "out_of_range": bool(np.any((action < -1.0) | (action > 1.0))),
                    "reward": float(reward),
                    "success": success,
                },
            )
            video.append(
                make_rollout_frame(
                    observation,
                    task_description=task_description,
                    step=steps,
                    replans=replans,
                    success=success,
                )
            )
            if success:
                termination = "success"
                break
            if terminated:
                termination = "terminated"
                break
            if truncated:
                termination = "truncated"
                break

    duration = time.monotonic() - started
    return {
        "episode": episode_index,
        "init_state_id": init_state_id,
        "seed": seed,
        "success": success,
        "termination": termination,
        "steps": steps,
        "replans": replans,
        "duration_s": duration,
        "inference": {
            "count": len(inference_latencies),
            "total_s": float(sum(inference_latencies)),
            "mean_s": float(np.mean(inference_latencies)) if inference_latencies else None,
            "max_s": float(max(inference_latencies)) if inference_latencies else None,
        },
        "action_latency": {
            "mean_s": float(np.mean(action_latencies)) if action_latencies else None,
            "max_s": float(max(action_latencies)) if action_latencies else None,
        },
        "action": {
            "all_finite": True,
            "min_per_dim": action_min.tolist() if steps else None,
            "max_per_dim": action_max.tolist() if steps else None,
        },
        "video_path": str(video_path),
    }


def _package_version(name: str) -> str | None:
    return importlib.metadata.version(name) if importlib.util.find_spec(name) else None


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _validate_positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparator}, got {value}")
    return value


def validate_libero_task(suite: Any, suite_name: str, task_id: int) -> Any:
    tasks = getattr(suite, "tasks", None)
    if not tasks:
        raise ValueError(f"LIBERO suite {suite_name!r} contains no tasks")
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(
            f"--task-id={task_id} is outside suite {suite_name!r} range "
            f"[0, {len(tasks) - 1}]"
        )
    return suite.get_task(task_id)


def _runtime_snapshot(args: Any, policy_config: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    return {
        "evaluation": {
            "suite": args.suite,
            "task_id": args.task_id,
            "episodes": args.episodes,
            "init_state_id": args.init_state_id,
            "seed": args.seed,
            "episode_length": args.episode_length,
            "num_steps_wait": args.num_steps_wait,
            "control_freq": args.control_freq,
            "observation_size": args.observation_size,
        },
        "libero": paths,
        "policy": policy_config,
        "environment": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }


class _EvaluationRun:
    """管理单次评测目录，并在异常时固化本次运行结果。"""

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.started = 0.0
        self.run_dir: Path
        self.result_path: Path
        self.actions_path: Path
        self.result: dict[str, Any]

    def __enter__(self) -> _EvaluationRun:
        self.started = time.monotonic()
        self.run_dir = _make_run_dir(self.output_root)
        _configure_logging(self.run_dir)
        self.result_path = self.run_dir / "result.json"
        self.actions_path = self.run_dir / "actions.jsonl"
        self.actions_path.touch()
        self.result = {
            "schema_version": 1,
            "status": "running",
            "start_time": datetime.now().astimezone().isoformat(),
            "output_dir": str(self.run_dir),
            "artifacts": {
                "result": str(self.result_path),
                "actions": str(self.actions_path),
                "log": str(self.run_dir / "run.log"),
                "runtime_config": str(self.run_dir / "runtime_config.yaml"),
            },
            "episodes": [],
        }
        self.write_result()
        return self

    def write_result(self) -> None:
        _write_json_atomic(self.result_path, self.result)

    def __exit__(self, exc_type: Any, exc_value: BaseException | None, tb: Any) -> bool:
        if exc_value is not None:
            self.result.update(
                {
                    "status": "error",
                    "duration_s": time.monotonic() - self.started,
                    "end_time": datetime.now().astimezone().isoformat(),
                    "error": {
                        "type": exc_type.__name__ if exc_type else type(exc_value).__name__,
                        "message": str(exc_value),
                        "traceback": "".join(traceback.format_exception(exc_type, exc_value, tb)),
                    },
                }
            )
            self.write_result()
        return False


def run_libero_eval(args: Any) -> int:
    """运行 LIBERO 评估，并管理本次运行的结果产物。"""
    with _EvaluationRun(args.output_dir) as evaluation:
        return _run_libero_eval(args, evaluation)


def _run_libero_eval(args: Any, evaluation: _EvaluationRun) -> int:
    """运行通用同步策略评估，并保存视频、动作日志和结果。"""
    run_dir = evaluation.run_dir
    actions_path = evaluation.actions_path
    result = evaluation.result
    _validate_positive(args.episodes, "--episodes")
    _validate_positive(args.init_state_id, "--init-state-id", allow_zero=True)
    _validate_positive(args.episode_length, "--episode-length")
    _validate_positive(args.num_steps_wait, "--num-steps-wait", allow_zero=True)
    _validate_positive(args.control_freq, "--control-freq")
    _validate_positive(args.observation_size, "--observation-size")

    paths = prepare_libero_runtime(args.libero_root, run_dir)
    libero_module = import_local_libero(args.libero_root)
    paths["import_origin"] = str(validate_libero_module_origin(libero_module, args.libero_root))
    logger.info("使用本地 LIBERO 源码：%s", paths["import_origin"])
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(str(args.suite))
    task_id = int(args.task_id)
    task = validate_libero_task(suite, str(args.suite), task_id)
    available_init_states = len(importlib.import_module("lerobot.envs.libero").get_task_init_states(suite, task_id))
    if args.init_state_id + args.episodes > available_init_states:
        raise ValueError(
            "Requested initial states exceed available LIBERO states: "
            f"start={args.init_state_id}, episodes={args.episodes}, available={available_init_states}"
        )
    task_description = str(task.language)

    config = load_config(config_path=args.config)
    config.policy["instruction"] = task_description
    runtime_policy_config = config.runtime_policy_config()
    _write_yaml_atomic(run_dir / "runtime_config.yaml", _runtime_snapshot(args, runtime_policy_config, paths))
    policy_type = normalize_policy_type(str(config.policy["type"]))
    result.update({
        "task": {"suite": args.suite, "task_id": task_id, "name": task.name, "instruction": task_description},
        "libero": {"root": paths["libero_root"], "import_origin": paths["import_origin"],
                    "git_revision": _git_revision(Path(paths["libero_root"])),
                    "available_init_states": available_init_states},
        "policy": {"type": policy_type, "pretrained_path": runtime_policy_config.get("pretrained_path"),
                    "device": runtime_policy_config.get("device"), "torch_dtype": runtime_policy_config.get("torch_dtype"),
                    "n_action_steps": runtime_policy_config.get("n_action_steps"),
                    "num_inference_timesteps": runtime_policy_config.get("num_inference_timesteps")},
        "versions": {name: _package_version(name) for name in ("lerobot", "torch", "mujoco", "robosuite")},
    })
    evaluation.write_result()

    policy = create_policy_adapter(runtime_policy_config)
    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=str(args.suite),
        episode_length=args.episode_length,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        init_states=True,
        episode_index=args.init_state_id,
        n_envs=1,
        num_steps_wait=args.num_steps_wait,
        control_freq=args.control_freq,
        control_mode="relative",
        hard_reset=True,
    )
    with ExitStack() as resources:
        resources.callback(env.close)
        with actions_path.open("w", encoding="utf-8") as action_log:
            for episode_index in range(args.episodes):
                init_state_id = args.init_state_id + episode_index
                video_path = (
                    run_dir / "rollout.mp4"
                    if args.episodes == 1
                    else run_dir / f"rollout_episode_{episode_index:03d}.mp4"
                )
                logger.info("开始 episode=%d init_state_id=%d", episode_index, init_state_id)
                result["episodes"].append(
                    run_single_episode(
                        env=env,
                        policy=policy,
                        task_description=task_description,
                        episode_index=episode_index,
                        init_state_id=init_state_id,
                        seed=args.seed + episode_index,
                        episode_length=args.episode_length,
                        control_freq=args.control_freq,
                        video_path=video_path,
                        action_log=action_log,
                    )
                )
                evaluation.write_result()

    successes = sum(bool(item["success"]) for item in result["episodes"])
    result["aggregate"] = {"successes": successes, "total_episodes": args.episodes,
                           "success_rate": successes / args.episodes,
                           "total_steps": sum(int(item["steps"]) for item in result["episodes"]),
                           "total_replans": sum(int(item["replans"]) for item in result["episodes"])}
    result["status"] = "completed"
    result["duration_s"] = time.monotonic() - evaluation.started
    result["end_time"] = datetime.now().astimezone().isoformat()
    evaluation.write_result()
    logger.info("LIBERO 评估完成：%d/%d 成功；产物=%s", successes, args.episodes, run_dir)
    return 0
