#!/usr/bin/env python3
"""Utility to run training, collect zero-style recordings, and retrain."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import random
import subprocess
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - rich is optional
    from rich.console import Console
    from rich.markdown import Markdown

    _HAS_RICH = True
except ImportError:  # pragma: no cover - graceful fallback when rich unavailable
    Console = None  # type: ignore
    Markdown = None  # type: ignore
    _HAS_RICH = False


INFO_STYLE = "cyan"
WARNING_STYLE = "yellow"
SUCCESS_STYLE = "green"
ERROR_STYLE = "bold red"
HEADER_STYLE = "bold blue"
DETAIL_STYLE = "dim"

if _HAS_RICH:
    CONSOLE = Console()
else:
    CONSOLE = None  # type: ignore


def _emit(message: str, *, style: Optional[str] = None, markdown: bool = False) -> None:
    if _HAS_RICH:
        if markdown:
            CONSOLE.print(Markdown(message), style=style)
        else:
            CONSOLE.print(message, style=style)
    else:
        print(message)


def _heading(title: str) -> None:
    _emit(
        f"### {title}" if _HAS_RICH else f"\n>>> {title}",
        style=HEADER_STYLE,
        markdown=_HAS_RICH,
    )


def _status(message: str, *, style: str = INFO_STYLE) -> None:
    _emit(message, style=style)


def _detail(message: str) -> None:
    _emit(message, style=DETAIL_STYLE)


@dataclass
class EpisodeMeta:
    episode_id: str
    even_timesteps: List[int]
    max_timestep: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an initial training phase, then record zero-style anchor episodes, and finally run a follow-up training phase on the newly simulated episodes."
        )
    )
    parser.add_argument(
        "--record-count",
        type=int,
        default=0,
        help="Number of anchored zero-style recordings to launch (default: 0)",
    )
    parser.add_argument(
        "--restarts-per-sample",
        type=int,
        default=2,
        help="How many times to re-launch each sampled episode/timestep (default: 2)",
    )
    parser.add_argument(
        "--episodes-dir-initial",
        type=Path,
        default=Path("persistent-data/state-service/episodes"),
        help="Episodes directory for the first training run",
    )
    parser.add_argument(
        "--episodes-dir-final",
        type=Path,
        default=Path("persistent-data/state-service/shallow-zero-style-episodes"),
        help="Episodes directory for the second training run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./logs/_training_artifacts"),
        help="Checkpoint output directory to pass to the training CLI",
    )
    parser.add_argument(
        "--pv-dir",
        type=Path,
        default=Path("persistent-data/trainer/models"),
        help="Directory where final models should be persisted",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device argument for the training CLI (default: cpu)",
    )
    parser.add_argument(
        "--backbone",
        default="lg_transformer",
        help="Backbone argument for the training CLI (default: lg_transformer)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=40,
        help="Number of BC epochs for training (default: 40)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=60,
        help="Number of PPO epochs for training (default: 60)",
    )
    parser.add_argument(
        "--post-recording-ppo-epochs",
        type=int,
        default=None,
        help=(
            "PPO epochs for the post-recording training run; defaults to half of "
            "--ppo-epochs (rounded up) when omitted"
        ),
    )
    parser.add_argument(
        "--h-max",
        type=int,
        default=300,
        help="Maximum height fed to the training CLI (default: 300)",
    )
    parser.add_argument(
        "--w-max",
        type=int,
        default=160,
        help="Maximum width fed to the training CLI (default: 160)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=3003,
        help="Port to use when launching start_zero_style.py (default: 3003)",
    )
    parser.add_argument(
        "--start-script",
        type=Path,
        default=None,
        help="Path to start_zero_style.py (defaults to repository root)",
    )
    parser.add_argument(
        "--recording-prompt",
        action="store_true",
        help="Pause after each recording to wait for user confirmation",
    )
    parser.add_argument(
        "--max-assistant-actions",
        type=int,
        default=1,
        help="Number of assistant actions to allow per zero-style recording",
    )
    parser.add_argument(
        "--human-follow-up-actions",
        type=int,
        default=1,
        help="Number of human policy actions to take after assistant actions",
    )
    parser.add_argument(
        "--record-timeout-seconds",
        type=int,
        default=600,
        help="Maximum seconds to wait for each recording to complete",
    )
    parser.add_argument(
        "--assistant-noise-prob",
        type=float,
        default=0.25,
        help=(
            "Probability that an assistant action in a zero-style simulation will be replaced "
            "with a random exploratory action (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--assistant-noise-top-k",
        type=int,
        default=3,
        help=(
            "When exploration noise triggers, sample the assistant's action uniformly from the top-k "
            "(action, line) pairs predicted by the policy (default: 3)"
        ),
    )
    parser.add_argument(
        "--record-poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polls while waiting for a recording to finish",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to reproduce episode/timestep sampling",
    )
    parser.add_argument(
        "--state-service-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the state service (used to wait for tester completion)",
    )
    parser.add_argument(
        "--tester-wait-seconds",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for the state service to finish running solution tests",
    )
    parser.add_argument(
        "--tester-poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polls while waiting for solution tests to finish",
    )
    parser.add_argument(
        "--train-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to policy_models.cli.run_tasks",
    )
    return parser.parse_args()


def resolve_start_script_path(arg_value: Path | None) -> Path:
    if arg_value is not None:
        return arg_value.resolve()
    return (Path(__file__).resolve().parent / "start_zero_style.py").resolve()


def ensure_episode_directory(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Episodes directory not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Episodes path is not a directory: {resolved}")
    return resolved

def get_consumed_episodes_dir(episodes_dir: Path) -> Path:
    """Return the consumed-episodes directory path next to the episodes directory."""
    return episodes_dir.parent / "consumed-episodes"


def mark_episodes_as_consumed(episodes_dir: Path, episode_metas: List[EpisodeMeta]) -> None:
    """Move consumed episodes to the consumed-episodes directory."""
    if not episode_metas:
        return
    
    consumed_dir = get_consumed_episodes_dir(episodes_dir)
    consumed_dir.mkdir(parents=True, exist_ok=True)
    
    _heading("Marking episodes as consumed")
    for meta in episode_metas:
        src = episodes_dir / meta.episode_id
        dst = consumed_dir / meta.episode_id
        if src.exists() and src.is_dir():
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
                _detail(f"Moved consumed episode: {meta.episode_id}")
            except Exception as e:
                _status(f"Warning: Could not move episode {meta.episode_id}: {e}", style=WARNING_STYLE)
    
    _status(f"Marked {len(episode_metas)} episode(s) as consumed", style=SUCCESS_STYLE)


def discover_episode_metadata(episodes_dir: Path) -> List[EpisodeMeta]:
    metas: List[EpisodeMeta] = []
        consumed_dir = get_consumed_episodes_dir(episodes_dir)
    consumed_ids = set()
    if consumed_dir.exists():
        for child in consumed_dir.iterdir():
            if child.is_dir():
                consumed_ids.add(child.name)
    
    for child in sorted(episodes_dir.iterdir()):
        if not child.is_dir():
            continue

                # Skip if this episode has already been consumed
        if child.name in consumed_ids:
            _detail(f"Skipping consumed episode: {child.name}")
            continue
        json_files = sorted(child.glob("*.json"))
        if not json_files:
            continue
        json_path = json_files[0]
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        states = payload.get("states")
        if not isinstance(states, list) or len(states) < 2:
            continue
        timesteps = [
            entry.get("timestep")
            for entry in states
            if isinstance(entry, dict) and isinstance(entry.get("timestep"), int)
        ]
        if not timesteps:
            continue
        max_timestep = max(timesteps)
        if max_timestep < 2:
            continue
        # Choose even timesteps within [0, max_timestep] but skip 0 since the UI
        # treats 0 as falsy and refuses to load the episode/timestep pair.
        even_timesteps = [step for step in range(0, max_timestep + 1, 2) if step != 0]
        if not even_timesteps:
            continue
        episode_id = payload.get("episode_id") or child.name
        metas.append(
            EpisodeMeta(
                episode_id=episode_id,
                even_timesteps=even_timesteps,
                max_timestep=max_timestep,
            )
        )
    if not metas:
        raise RuntimeError(
            f"No eligible episodes with JSON logs were found under {episodes_dir}"
        )
    return metas


def _build_env_with_pythonpath(extra_paths: List[Path]) -> dict:
    env = os.environ.copy()
    extras = [str(p) for p in extra_paths if p]
    if extras:
        extra_str = os.pathsep.join(extras)
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{extra_str}{os.pathsep}{current}" if current else extra_str
        )
    return env


def _read_tail(path: Path, max_bytes: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return ""
            read = min(size, max_bytes)
            handle.seek(-read, os.SEEK_END)
            return handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _completed_episode_ids(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    completed: set[str] = set()
    for episode_dir in directory.iterdir():
        if not episode_dir.is_dir():
            continue
        raw_file = episode_dir / "raw" / f"{episode_dir.name}.jsonl"
        if not raw_file.exists():
            continue
        tail = _read_tail(raw_file)
        if '"endTime"' in tail:
            completed.add(episode_dir.name)
    return completed


def _wait_for_zero_style_tests(
    state_service_url: str,
    timeout: float,
    poll_interval: float,
) -> None:
    """Block until the state service's tester queue drains for zero-style episodes."""

    parsed = urllib.parse.urlparse(state_service_url)
    if not parsed.scheme:
        raise ValueError(
            "--state-service-url must include a scheme, e.g., http://localhost:8000"
        )

    status_url = urllib.parse.urljoin(
        state_service_url.rstrip("/") + "/", "test-queue/status"
    )
    # Ensure zero-style store is selected
    query = urllib.parse.urlencode({"zerostyle": "true"})
    status_url = f"{status_url}?{query}"

    _heading("Waiting for state service tester queue to drain (zero-style)")
    deadline = time.time() + timeout
    last_queue_size: float | None = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                status_url, timeout=poll_interval + 2.0
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            _status(
                f"Failed to poll tester queue status: {exc}",
                style=WARNING_STYLE,
            )
            time.sleep(poll_interval)
            continue

        queue_available = bool(payload.get("queue_available"))
        is_empty = bool(payload.get("is_empty"))
        queue_size = payload.get("queue_size")

        if queue_size != last_queue_size:
            if queue_available:
                _detail(f"tester queue size={queue_size}; is_empty={is_empty}")
            else:
                _status("tester queue unavailable; waiting", style=WARNING_STYLE)
            last_queue_size = queue_size

        if queue_available and is_empty:
            _status("Tester queue drained; continuing", style=SUCCESS_STYLE)
            return

        time.sleep(poll_interval)

    raise TimeoutError(
        "Timed out waiting for the state service tester queue to drain; zero-style episodes may "
        "lack complete test metadata."
    )


def _await_new_episode(
    directory: Path,
    baseline: set[str],
    timeout: float,
    poll_interval: float,
) -> set[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = _completed_episode_ids(directory)
        if current - baseline:
            return current
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for a new episode under {directory} after {timeout} seconds"
    )


def run_training(
    episodes_dir: Path,
    checkpoint_root: Path,
    pv_dir: Path,
    device: str,
    backbone: str,
    bc_epochs: int,
    ppo_epochs: int,
    h_max: int,
    w_max: int,
    extra_args: List[str],
    pythonpath: List[Path],
    init_from_pv: bool = False,
    tb_root: Optional[Path] = None,
    run_label: str = "",
) -> None:
    run_label = run_label.strip()
    phase_checkpoint_dir = (
        checkpoint_root / run_label if run_label else checkpoint_root
    ).resolve()
    phase_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_dir: Optional[Path] = None
    if tb_root is not None:
        tb_dir = (tb_root / run_label) if run_label else tb_root
        tb_dir = tb_dir.resolve()
        tb_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "policy_models.cli.run_tasks",
        "train_from_episodes",
        "--episodes_dir",
        str(episodes_dir),
        "--checkpoint_dir",
        str(phase_checkpoint_dir),
        "--pv_dir",
        str(pv_dir),
        "--device",
        device,
        "--backbone",
        backbone,
        "--bc_epochs",
        str(bc_epochs),
        "--ppo_epochs",
        str(ppo_epochs),
        "--h_max",
        str(h_max),
        "--w_max",
        str(w_max),
    ]
    if tb_dir is not None:
        cmd.extend(["--tb_dir", str(tb_dir)])
    if run_label:
        cmd.extend(["--run_label", run_label])
    if extra_args:
        cmd.extend(extra_args)
    if init_from_pv:
        cmd.append("--init-from-pv")

    _heading("Running training command")
    _detail(" ".join(cmd))
    env = _build_env_with_pythonpath(pythonpath)
    subprocess.run(cmd, check=True, env=env)


def launch_recordings(
    start_script: Path,
    episodes_dir: Path,
    output_dir: Path,
    metas: List[EpisodeMeta],
    record_count: int,
    start_port: int,
    restarts_per_sample: int,
    recording_prompt: bool,
    max_assistant_actions: int,
    max_human_actions: int,
    assistant_noise_prob: float,
    assistant_noise_top_k: int,
    record_timeout: float,
    poll_interval: float,
    pythonpath: List[Path],
) -> None:
    if record_count <= 0:
        _status(
            "Skipping zero-style recordings: record count <= 0", style=WARNING_STYLE
        )
        return

    _heading("Launching anchored zero-style recordings")
    known_snapshots = _completed_episode_ids(output_dir)
    for idx in range(record_count):
        meta = random.choice(metas)
        timestep = random.choice(meta.even_timesteps)
        cmd = [
            sys.executable,
            str(start_script),
            "--episode",
            meta.episode_id,
            "--timestep",
            str(timestep),
            "--episodes-dir",
            str(episodes_dir),
            "--port",
            str(start_port),
            "--max-assistant-actions",
            str(max_assistant_actions),
            "--max-human-actions",
            str(max_human_actions),
            "--assistant-noise-prob",
            str(assistant_noise_prob),
            "--assistant-noise-top-k",
            str(assistant_noise_top_k),
        ]
        _status(
            f"[{idx + 1}/{record_count}] episode={meta.episode_id} "
            f"timestep={timestep} (max timestep={meta.max_timestep})"
        )
        for restart_idx in range(restarts_per_sample):
            if restarts_per_sample > 1:
                _detail(f"  restart {restart_idx + 1}/{restarts_per_sample}")
            env = _build_env_with_pythonpath(pythonpath)
            subprocess.run(cmd, check=True, env=env)
            try:
                known_snapshots = _await_new_episode(
                    output_dir,
                    known_snapshots,
                    record_timeout,
                    poll_interval,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "Anchored zero-style recording did not finish in time; "
                    "no new episodes were detected."
                ) from exc
            if recording_prompt:
                input("Press Enter after finishing this recording to continue...")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    start_script = resolve_start_script_path(args.start_script)
    if not start_script.exists():
        raise FileNotFoundError(f"start_zero_style.py not found at: {start_script}")

    if args.restarts_per_sample <= 0:
        raise ValueError("--restarts-per-sample must be a positive integer")
    if args.max_assistant_actions <= 0:
        raise ValueError("--max-assistant-actions must be a positive integer")
    if args.human_follow_up_actions < 0:
        raise ValueError("--human-follow-up-actions must be zero or a positive integer")
    if args.record_timeout_seconds <= 0:
        raise ValueError("--record-timeout-seconds must be a positive integer")
    if args.record_poll_interval <= 0:
        raise ValueError("--record-poll-interval must be a positive number")
    if args.assistant_noise_top_k <= 0:
        raise ValueError("--assistant-noise-top-k must be a positive integer")
    if args.tester_wait_seconds <= 0:
        raise ValueError("--tester-wait-seconds must be positive")
    if args.tester_poll_interval <= 0:
        raise ValueError("--tester-poll-interval must be positive")

    episodes_dir_initial = ensure_episode_directory(args.episodes_dir_initial)
    episodes_dir_final = ensure_episode_directory(args.episodes_dir_final)
    checkpoint_root = args.checkpoint_dir.resolve()
    pv_dir = args.pv_dir.resolve()

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pv_dir.mkdir(parents=True, exist_ok=True)

    tb_root = (checkpoint_root / "tb").resolve()
    tb_root.mkdir(parents=True, exist_ok=True)

    metas = discover_episode_metadata(episodes_dir_initial)

    pythonpath_entries = [Path(__file__).resolve().parent / "policy_models"]
    pythonpath_entries = [p for p in pythonpath_entries if p.exists()]

    run_training(
        episodes_dir=episodes_dir_initial,
        checkpoint_root=checkpoint_root,
        pv_dir=pv_dir,
        device=args.device,
        backbone=args.backbone,
        bc_epochs=args.bc_epochs,
        ppo_epochs=args.ppo_epochs,
        h_max=args.h_max,
        w_max=args.w_max,
        extra_args=args.train_extra_args,
        pythonpath=pythonpath_entries,
        tb_root=tb_root,
        run_label="human",
    )

    launch_recordings(
        start_script=start_script,
        episodes_dir=episodes_dir_initial,
        output_dir=episodes_dir_final,
        metas=metas,
        record_count=args.record_count,
        start_port=args.start_port,
        restarts_per_sample=args.restarts_per_sample,
        recording_prompt=args.recording_prompt,
        max_assistant_actions=args.max_assistant_actions,
        max_human_actions=args.human_follow_up_actions,
        assistant_noise_prob=args.assistant_noise_prob,
        assistant_noise_top_k=args.assistant_noise_top_k,
        record_timeout=args.record_timeout_seconds,
        poll_interval=args.record_poll_interval,
        pythonpath=pythonpath_entries,
    )

    try:
        _wait_for_zero_style_tests(
            state_service_url=args.state_service_url,
            timeout=args.tester_wait_seconds,
            poll_interval=args.tester_poll_interval,
        )
    except TimeoutError as exc:
        _status(str(exc), style=WARNING_STYLE)
    except Exception as exc:  # pragma: no cover - defensive
        _status(
            f"Failed to confirm tester completion: {exc}",
            style=WARNING_STYLE,
        )

    post_recording_ppo_epochs = args.post_recording_ppo_epochs
    if post_recording_ppo_epochs is None:
        post_recording_ppo_epochs = max(1, math.ceil(args.ppo_epochs / 2))

    run_training(
        episodes_dir=episodes_dir_final,
        checkpoint_root=checkpoint_root,
        pv_dir=pv_dir,
        device=args.device,
        backbone=args.backbone,
        bc_epochs=0,
        ppo_epochs=post_recording_ppo_epochs,
        h_max=args.h_max,
        w_max=args.w_max,
        extra_args=args.train_extra_args,
        pythonpath=pythonpath_entries,
        init_from_pv=True,
        tb_root=tb_root,
        run_label="zero_style",
    )


if __name__ == "__main__":
    main()
