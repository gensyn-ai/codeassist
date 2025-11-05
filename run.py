from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

import argparse
import io
import json
import logging
import time
import os
import webbrowser as browser
import shutil

from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)

import docker
import requests
import subprocess
import sys

from huggingface_hub import HfApi, login, whoami

from ca_alchemy import contract_caller

# Rich console for colored output
CONSOLE = Console()
LOG_COLOR = "dim"
HEADER_COLOR = "bold blue"
INFO_COLOR = "cyan"
ERROR_COLOR = "bold red"
WARNING_COLOR = "yellow"
SUCCESS_COLOR = "green"
GENSYN_COLOR = "bold magenta"

# Retrieve the user's HF_TOKEN from the environment
HF_TOKEN = os.environ.get("HF_TOKEN")

DOCKER_CLIENT = None

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/run.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

PERSISTENT_DATA_DIR = Path("./persistent-data")


@dataclass
class Config:
    branch: str
    ollama_tag: str
    no_telemetry: bool
    no_pull: bool
    no_upload: bool
    network_name: str
    no_train: bool
    training_config_path: str
    no_sc: bool
    train_only: bool


def detect_docker():
    """Detect if Docker is installed and running."""
    global DOCKER_CLIENT

    CONSOLE.print("Looking for a container engine...", style=INFO_COLOR)

    try:
        logger.info("Checking for Docker...")
        DOCKER_CLIENT = docker.from_env()
        DOCKER_CLIENT.ping()
        logger.info("Docker is installed and running.")
        return True
    except docker.errors.DockerException as e:
        logger.info(f"Docker not found or not running: {e}")

    # Check for Colima (macOS/Linux)
    try:
        logger.info("Checking for Colima...")
        home = os.path.expanduser("~")
        DOCKER_CLIENT = docker.from_env(
            environment={"DOCKER_HOST": f"unix://{home}/.colima/default/docker.sock"}
        )
        DOCKER_CLIENT.ping()
        logger.info("Colima is installed and running.")
        return True
    except docker.errors.DockerException as e:
        logger.info(f"Colima not found or not running: {e}")

    return False


def cleanup_incomplete_episodes(episodes_root: Path) -> None:
    """Remove episode directories missing their finalized snapshot JSON."""

    if not episodes_root.exists():
        return

    pruned = 0
    for episode_dir in episodes_root.iterdir():
        if not episode_dir.is_dir():
            continue

        marker = episode_dir / f"{episode_dir.name}.json"
        if marker.exists():
            continue

        try:
            shutil.rmtree(episode_dir)
            pruned += 1
        except OSError as exc:
            logger.warning(
                "Failed to remove incomplete episode directory %s: %s",
                episode_dir,
                exc,
            )

    if pruned:
        message = (
            f"Removed {pruned} incomplete episode directory"
            f"{'ies' if pruned != 1 else ''} from {episodes_root}."
        )
        CONSOLE.print(message, style=INFO_COLOR)
        logger.info(message)


def wait_for_healthy(container: docker.models.containers.Container):
    """Wait for a Docker container to become healthy."""
    global DOCKER_CLIENT
    logger.info(f"Waiting for container {container.name} to become healthy...")
    while True:
        try:
            container.reload()  # Refresh container attributes
            status = container.attrs.get("State", {}).get("Health", {}).get("Status")
            if status == "healthy":
                logger.info(f"Container {container.name} is healthy.")
                CONSOLE.print(
                    f"Container {container.name} is healthy.", style=LOG_COLOR
                )
                break
            elif status == "unhealthy":
                logger.error(f"Container {container.name} is unhealthy.")
                raise Exception(f"Container {container.name} is unhealthy.")
            else:
                logger.info(f"Container {container.name} status: {status}. Waiting...")
        except docker.errors.NotFound:
            logger.error(f"Container {container.name} not found.")
            raise
        time.sleep(15)  # Wait before checking again


def wait_for_http_service(url: str, name: str, timeout: int = 120, interval: int = 5):
    """Wait until an HTTP endpoint responds with any status code."""

    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=interval)
            logger.info("%s responded with status %s", name, response.status_code)
            return
        except requests.RequestException as exc:
            last_error = exc
            logger.info("Waiting for %s at %s: %s", name, url, exc)
            time.sleep(interval)

    raise TimeoutError(
        f"Timed out waiting for {name} to become reachable at {url}: {last_error}"
    )


def setup_persistent_volume():
    """Set up the persistent volume directory structure."""

    # Create main data directory
    PERSISTENT_DATA_DIR.mkdir(exist_ok=True)

    # Create subdirectories for each component
    directories = [
        "state-service/episodes",  # State service episode data
        "state-service/simulated-episodes",  # State service simulated episode data
        "state-service/shallow-zero-style-episodes",  # Zero-style anchor recordings
        "solution-tester/results",  # Test results and reports
        "trainer/models",  # Trained ASM model checkpoints
    ]

    for directory in directories:
        dir_path = PERSISTENT_DATA_DIR / directory
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy asm_assistant_model.pt and asm_featurizer.pt to trainer/models if they don't exist in the directory
    baseline_dir = Path("policy_models/baseline")
    trainer_models_dir = PERSISTENT_DATA_DIR / "trainer/models"

    model_files = ["asm_assistant_model.pt", "asm_featurizer.pt", "asm_human_model.pt"]

    for model_file in model_files:
        source_path = baseline_dir / model_file
        dest_path = trainer_models_dir / model_file

        if source_path.exists() and not dest_path.exists():
            logger.info(f"Copying {model_file} from baseline to trainer/models...")
            shutil.copy2(source_path, dest_path)
            logger.info(f"Successfully copied {model_file}")
        elif dest_path.exists():
            logger.info(
                f"{model_file} exists in trainer/models directory, using file for inference"
            )

    subprocess.run(f"chmod -R 777 {PERSISTENT_DATA_DIR}", shell=True)

    logger.info(
        f"Persistent volume structure created at {PERSISTENT_DATA_DIR.absolute()}"
    )


def ensure_network(config: Config) -> docker.models.networks.Network:
    global DOCKER_CLIENT
    logger.info(f"Ensuring network '{config.network_name}' exists...")
    try:
        network = DOCKER_CLIENT.networks.get(config.network_name)
    except docker.errors.NotFound:
        network = DOCKER_CLIENT.networks.create(config.network_name)
    return network


def setup_ollama(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"ollama/ollama:{config.ollama_tag}"

    CONSOLE.print("Setting up Ollama...", style=LOG_COLOR)

    logger.info("Checking for existing Ollama containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-ollama").remove(force=True)
        logger.info("Removed existing Ollama container.")
    except docker.errors.NotFound:
        logger.info("No existing Ollama container found.")

    if not config.no_pull:
        logger.info(f"Pulling Ollama image at tag {config.ollama_tag}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("Ollama image pulled successfully.")

    logger.info("Starting Ollama container...")

    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-ollama",
        ports={
            "11434/tcp": 11434,  # Expose Ollama API port
        },
        volumes={
            f"{os.getcwd()}/ollama-data": {"bind": "/root/.ollama", "mode": "rw"},
        },
    )

    logger.info(f"Ollama container started with ID: {container.id}")

    CONSOLE.print("Ollama started", style=LOG_COLOR)

    return container


def setup_web_ui(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"gensynai/codeassist-web-ui:{config.branch}"

    CONSOLE.print("Setting up Web UI...", style=LOG_COLOR)

    logger.info("Checking for existing Web UI containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-web-ui").remove(force=True)
        logger.info("Removed existing Web UI container.")
    except docker.errors.NotFound:
        logger.info("No existing Web UI container found.")

    if not config.no_pull:
        logger.info(f"Pulling Web UI image at tag {config.branch}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("Web UI image pulled successfully.")

    logger.info("Starting Web UI container...")
    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-web-ui",
        ports={
            "3000/tcp": 3000,  # Expose Web UI port
        },
        volumes={
            f"{os.getcwd()}/persistent-data": {
                "bind": "/app/persistent-data",
                "mode": "rw",
            },
        },
    )

    logger.info(f"Web UI container started with ID: {container.id}")

    CONSOLE.print("Web UI started", style=LOG_COLOR)

    return container


def setup_state_service(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"gensynai/codeassist-state-service:{config.branch}"

    CONSOLE.print("Setting up State Service...", style=LOG_COLOR)

    logger.info("Checking for existing State Service containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-state-service").remove(force=True)
        logger.info("Removed existing State Service container.")
    except docker.errors.NotFound:
        logger.info("No existing State Service container found.")

    if not config.no_pull:
        logger.info(f"Pulling State Service image at tag {config.branch}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("State Service image pulled successfully.")

    logger.info("Starting State Service container...")
    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-state-service",
        ports={
            "8000/tcp": 8000,  # Expose State Service port
        },
        environment={
            "OLLAMA_BASE_URL": "http://codeassist-ollama:11434",
            "OLLAMA_HOST": "http://codeassist-ollama:11434",
            "PERSISTENT_DATA_DIR": "/app/persistent-data",
            "SOLUTION_TESTER_BASE_URL": "http://codeassist-solution-tester:8008",
            "POLICY_MODEL_BASE_URL": "http://codeassist-policy-model:8001",
            "TELEMETRY_BASE_URL": "https://telemetry-api.internal-apps-central1.clusters.gensyn.ai",
            "DISABLE_TELEMETRY": "true" if config.no_telemetry else "false",
        },
        volumes={
            f"{os.getcwd()}/persistent-data": {
                "bind": "/app/persistent-data",
                "mode": "rw",
            },
            f"{os.getcwd()}/datasets": {"bind": "/app/datasets", "mode": "ro"},
        },
    )

    logger.info(f"State Service container started with ID: {container.id}")
    CONSOLE.print("State Service started", style=LOG_COLOR)

    return container


def setup_solution_tester(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"gensynai/codeassist-solution-tester:{config.branch}"

    CONSOLE.print("Setting up Solution Tester...", style=LOG_COLOR)

    logger.info("Checking for existing Solution Tester containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-solution-tester").remove(force=True)
        logger.info("Removed existing Solution Tester container.")
    except docker.errors.NotFound:
        logger.info("No existing Solution Tester container found.")

    if not config.no_pull:
        logger.info(f"Pulling Solution Tester image at tag {config.branch}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("Solution Tester image pulled successfully.")

    logger.info("Starting Solution Tester container...")
    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-solution-tester",
        ports={
            "8008/tcp": 8008,  # Expose Solution Tester port
        },
        volumes={
            f"{os.getcwd()}/persistent-data": {
                "bind": "/app/persistent-data",
                "mode": "rw",
            },
        },
    )

    logger.info(f"Solution Tester container started with ID: {container.id}")
    CONSOLE.print("Solution Tester started", style=LOG_COLOR)

    return container


def setup_policy_models(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"gensynai/codeassist-policy-model:{config.branch}"

    CONSOLE.print("Setting up Policy Models...", style=LOG_COLOR)

    logger.info("Checking for existing Policy Models containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-policy-model").remove(force=True)
        logger.info("Removed existing Policy Models container.")
    except docker.errors.NotFound:
        logger.info("No existing Policy Models container found.")

    if not config.no_pull:
        logger.info(f"Pulling Policy Models image at tag {config.branch}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("Policy Models image pulled successfully.")

    logger.info("Starting Policy Models container...")
    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-policy-model",
        ports={
            "8001/tcp": 8001,  # Expose Policy Models API port
        },
        environment={
            "DEVICE": "cpu",
            "OLLAMA_BASE_URL": "http://codeassist-ollama:11434",
            "OLLAMA_HOST": "http://codeassist-ollama:11434",
            "PERSISTENT_DATA_DIR": "/app/persistent-data",
            "ASM_ASSISTANT_MODEL_PATH": "/app/persistent-data/trainer/models/asm_assistant_model.pt",
            "ASM_FEATURIZER_PATH": "/app/persistent-data/trainer/models/asm_featurizer.pt",
            "TELEMETRY_BASE_URL": "https://telemetry-api.internal-apps-central1.clusters.gensyn.ai",
            "DISABLE_TELEMETRY": "true" if config.no_telemetry else "false",
        },
        volumes={
            f"{os.getcwd()}/persistent-data": {
                "bind": "/app/persistent-data",
                "mode": "rw",
            },
        },
    )

    logger.info(f"Policy Models container started with ID: {container.id}")
    CONSOLE.print("Policy Models started", style=LOG_COLOR)

    return container


def setup_zero_style_ui(config: Config) -> docker.models.containers.Container:
    global DOCKER_CLIENT
    image = f"gensynai/codeassist-zero-style-ui:{config.branch}"

    CONSOLE.print("Setting up Zero-style UI...", style=LOG_COLOR)

    logger.info("Checking for existing Zero-style UI containers...")
    try:
        DOCKER_CLIENT.containers.get("codeassist-zero-style-ui").remove(force=True)
        logger.info("Removed existing Zero-style UI container.")
    except docker.errors.NotFound:
        logger.info("No existing Zero-style UI container found.")

    if not config.no_pull:
        logger.info(f"Pulling Zero-style UI image at tag {config.branch}...")
        DOCKER_CLIENT.images.pull(image)
        logger.info("Zero-style UI image pulled successfully.")

    logger.info("Starting Zero-style UI container...")
    container = DOCKER_CLIENT.containers.run(
        image,
        detach=True,
        network=config.network_name,
        auto_remove=False,
        name="codeassist-zero-style-ui",
        ports={
            "3000/tcp": 3003,  # Expose Zero-style UI port
        },
        environment={
            "PERSISTENT_DATA_DIR": "/app/persistent-data",
        },
        volumes={
            f"{os.getcwd()}/persistent-data": {
                "bind": "/app/persistent-data",
                "mode": "rw",
            },
        },
    )

    logger.info(f"Zero-style UI container started with ID: {container.id}")
    CONSOLE.print("Zero-style UI started", style=LOG_COLOR)

    try:
        wait_for_http_service("http://localhost:3003", "Zero-style UI")
        CONSOLE.print(
            "Zero-style UI is reachable at http://localhost:3003", style=LOG_COLOR
        )
    except TimeoutError as exc:
        logger.warning(str(exc))
        CONSOLE.print(
            "Zero-style UI did not become reachable in time. Training may fail to launch recordings.",
            style=WARNING_COLOR,
        )

    return container


def stop_all_containers(containers):
    """Stop all Docker containers with error handling."""
    CONSOLE.print("Stopping all containers...", style=INFO_COLOR)
    logger.info("Stopping all containers...")

    for container_name, container in containers.items():
        try:
            if container:
                container.stop()
                logger.info(f"Stopped {container_name}")
        except Exception as e:
            logger.warning(f"Failed to stop {container_name}: {e}")

    CONSOLE.print("All containers stopped", style=INFO_COLOR)
    logger.info("All containers stopped")


def cleanup_zero_style_recordings(episodes_dir: Path) -> None:
    """Remove anchored zero-style recordings to keep disk usage in check."""
    try:
        episodes_dir = episodes_dir.expanduser().resolve(strict=False)

        if not episodes_dir.exists():
            logger.info(
                "Zero-style recordings directory %s not found; nothing to clean up.",
                episodes_dir,
            )
            return

        entries = list(episodes_dir.iterdir())
        if not entries:
            logger.info(
                "Zero-style recordings directory %s already empty.", episodes_dir
            )
            return

        removed = 0
        for entry in entries:
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to remove zero-style recording artifact %s: %s",
                    entry,
                    exc,
                )

        if removed:
            logger.info(
                "Removed %s anchored zero-style recording artifact(s) from %s",
                removed,
                episodes_dir,
            )
            CONSOLE.print(
                f"Removed {removed} anchored zero-style recording artifact(s) from"
                f" {episodes_dir}",
                style=LOG_COLOR,
            )
        else:
            logger.info(
                "No anchored zero-style recording artifacts removed from %s",
                episodes_dir,
            )

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to clean anchored zero-style recordings in %s: %s",
            episodes_dir,
            exc,
        )
        CONSOLE.print(
            f"Failed to clean anchored zero-style recordings in {episodes_dir}: {exc}",
            style=WARNING_COLOR,
        )


def run_training(config: Config) -> bool:
    """Run the training loop locally using the JSON configuration."""
    CONSOLE.print("Starting policy model training...", style=LOG_COLOR)
    logger.info("Starting policy model training...")

    repo_root = Path(__file__).resolve().parent

    config_path = Path(config.training_config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        logger.error(f"Training config file not found: {config.training_config_path}")
        CONSOLE.print(
            f"Training config file not found: {config.training_config_path}",
            style=ERROR_COLOR,
        )
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            training_config = json.load(handle)
        logger.info(f"Loaded training config from {config_path}")
    except Exception as exc:  # pragma: no cover - defensive file handling
        logger.error(f"Failed to load training config: {exc}")
        CONSOLE.print(f"Failed to load training config: {exc}", style=ERROR_COLOR)
        return False

    default_cleanup_dir = (
        repo_root / PERSISTENT_DATA_DIR / "state-service/shallow-zero-style-episodes"
    ).resolve()
    cleanup_target = training_config.get("episodes_dir_final")
    if cleanup_target:
        cleanup_dir = Path(cleanup_target).expanduser()
        if not cleanup_dir.is_absolute():
            cleanup_dir = (repo_root / cleanup_dir).resolve()
        else:
            cleanup_dir = cleanup_dir.resolve()
    else:
        cleanup_dir = default_cleanup_dir

    option_map = {
        "record_count": "--record-count",
        "restarts_per_sample": "--restarts-per-sample",
        "episodes_dir_initial": "--episodes-dir-initial",
        "episodes_dir_final": "--episodes-dir-final",
        "checkpoint_dir": "--checkpoint-dir",
        "pv_dir": "--pv-dir",
        "device": "--device",
        "backbone": "--backbone",
        "bc_epochs": "--bc-epochs",
        "ppo_epochs": "--ppo-epochs",
        "post_recording_ppo_epochs": "--post-recording-ppo-epochs",
        "h_max": "--h-max",
        "w_max": "--w-max",
        "start_port": "--start-port",
        "start_script": "--start-script",
        "recording_prompt": "--recording-prompt",
        "max_assistant_actions": "--max-assistant-actions",
        "human_follow_up_actions": "--human-follow-up-actions",
        "record_timeout_seconds": "--record-timeout-seconds",
        "assistant_noise_prob": "--assistant-noise-prob",
        "assistant_noise_top_k": "--assistant-noise-top-k",
        "record_poll_interval": "--record-poll-interval",
        "seed": "--seed",
        "state_service_url": "--state-service-url",
        "tester_wait_seconds": "--tester-wait-seconds",
        "tester_poll_interval": "--tester-poll-interval",
    }

    script_path = repo_root / "training_loop.py"
    uv_executable = shutil.which("uv")

    if uv_executable:
        training_cmd = [uv_executable, "run", "python", "training_loop.py"]
    else:
        logger.warning(
            "'uv' command not found; falling back to host Python interpreter"
        )
        if not sys.executable:
            logger.error("Unable to locate Python interpreter for training")
            CONSOLE.print(
                "Unable to locate Python interpreter for training",
                style=ERROR_COLOR,
            )
            return False
        training_cmd = [sys.executable, str(script_path)]

    unknown_keys = set(training_config.keys()) - (
        set(option_map.keys()) | {"train_extra_args"}
    )
    if unknown_keys:
        logger.warning(
            "Unsupported training config options will be ignored: %s",
            ", ".join(sorted(str(key) for key in unknown_keys)),
        )

    for key, flag in option_map.items():
        if key not in training_config:
            continue
        value = training_config[key]
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            if value:
                training_cmd.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                training_cmd.extend([flag, str(item)])
            continue
        training_cmd.extend([flag, str(value)])

    extra_args = training_config.get("train_extra_args", [])
    if isinstance(extra_args, list):
        training_cmd.extend(str(arg) for arg in extra_args)
    elif extra_args:
        training_cmd.append(str(extra_args))

    logger.info(f"Running training command locally: {' '.join(training_cmd)}")
    CONSOLE.print(
        f"Running training command locally: {' '.join(training_cmd)}",
        style=LOG_COLOR,
    )

    ran_training_command = False
    try:
        env = os.environ.copy()
        env["TELEMETRY_BASE_URL"] = (
            "https://telemetry-api.internal-apps-central1.clusters.gensyn.ai"
        )
        env["DISABLE_TELEMETRY"] = "true" if config.no_telemetry else "false"
        result = subprocess.run(
            training_cmd,
            check=False,
            cwd=str(repo_root),
            env=env,
        )
        ran_training_command = True
    except FileNotFoundError as exc:
        logger.error(f"Failed to run training command: {exc}")
        CONSOLE.print(f"Failed to run training command: {exc}", style=ERROR_COLOR)
        return False
    finally:
        if ran_training_command:
            cleanup_zero_style_recordings(cleanup_dir)

    if result.returncode == 0:
        logger.info("Training completed successfully")
        CONSOLE.print("Training completed successfully", style=SUCCESS_COLOR)
        return True

    logger.error(f"Training failed with exit code {result.returncode}")
    CONSOLE.print(
        f"Training failed with exit code {result.returncode}",
        style=ERROR_COLOR,
    )
    return False


def await_testing_queue_completion():
    """Poll the testing queue status every 10 seconds until it's empty or max attempts reached."""
    CONSOLE.print("Polling testing queue status...", style=INFO_COLOR)

    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        try:
            # Make request to the test queue status endpoint
            response = requests.get(
                "http://localhost:8000/test-queue/status", timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                is_empty = data.get("is_empty", True)
                queue_size = data.get("queue_size", 0)

                if is_empty:
                    CONSOLE.print(
                        "Testing queue is empty. Proceeding to training.",
                        style=SUCCESS_COLOR,
                    )
                    logger.info("Testing queue is empty. Proceeding to training.")
                    return True
                else:
                    CONSOLE.print(
                        f"Testing queue has {queue_size} items remaining. Attempt {attempt}/{max_attempts}...",
                        style=INFO_COLOR,
                    )
                    logger.info(
                        f"Testing queue has {queue_size} items remaining. Attempt {attempt}/{max_attempts}..."
                    )
            else:
                logger.warning(
                    f"Failed to get queue status: HTTP {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error checking queue status: {e}")

        # Wait 10 seconds before next check
        time.sleep(10)

    # If we reach here, we've exceeded max attempts
    CONSOLE.print(
        f"Maximum attempts ({max_attempts}) reached. Proceeding to training anyway.",
        style=WARNING_COLOR,
    )
    logger.warning(
        f"Maximum attempts ({max_attempts}) reached. Proceeding to training anyway."
    )
    return False


def upload_to_huggingface(hf_token: str, folder_path: Path):
    logger.info("Uploading model data to HuggingFace...")
    CONSOLE.print("Uploading model data to HuggingFace...", style=INFO_COLOR)

    hf_api = HfApi(token=hf_token)

    username = whoami(token=hf_token)["name"]
    login(token=hf_token)

    repo_id = f"{username}/codeassist"

    hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    eoa = "0xD819f9B9a60EB9B9129bB35d9F2b848Bd6ca0B72"

    metadata_json = json.dumps({"eoa": eoa}, indent=4)
    metadata_bytes = io.BytesIO(metadata_json.encode("utf-8"))
    hf_api.upload_file(
        path_or_fileobj=metadata_bytes,
        path_in_repo="gensyn.json",
        repo_id=repo_id,
        repo_type="model",
    )

    commit_info = hf_api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=folder_path
    )
    hf_api.upload_file(
        path_or_fileobj="hf_readme.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    logger.info(
        f"Model data uploaded to HuggingFace at https://huggingface.co/{repo_id}"
    )

    if not config.no_sc:
        contract_success = contract_caller.submit_hf_upload(repo_id, commit_info.oid)

        if not contract_success:
            logger.error("Failed to submit HF upload to contract.")
            CONSOLE.print(
                "Failed to submit HF upload to contract. Check the logs for more details.",
                style=ERROR_COLOR,
            )
            return

        logger.info(
            f"Submitted HF upload to contract for repo {repo_id} ({commit_info.oid})"
        )

    CONSOLE.print(
        f"Model data uploaded to HuggingFace at https://huggingface.co/{repo_id}",
        style=SUCCESS_COLOR,
    )


def start_containers(config: Config):
    ollama_container = None
    policy_models_container = None
    web_ui_container = None
    state_service_container = None
    solution_tester_container = None
    with Progress(
        TextColumn("[cyan]Setting up containers..."),
        BarColumn(),
        MofNCompleteColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Setting up containers...", total=5)

        ollama_container = setup_ollama(config)
        progress.update(task, advance=1)

        policy_models_container = setup_policy_models(config)
        progress.update(task, advance=1)

        web_ui_container = setup_web_ui(config)
        progress.update(task, advance=1)

        state_service_container = setup_state_service(config)
        progress.update(task, advance=1)

        solution_tester_container = setup_solution_tester(config)
        progress.update(task, advance=1)

    with Progress(
        TextColumn("[cyan]Waiting for containers to come online..."),
        BarColumn(),
        MofNCompleteColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task(
            "[cyan]Waiting for containers to come online...", total=5
        )

        wait_for_healthy(policy_models_container)
        progress.update(task, advance=1)

        wait_for_healthy(web_ui_container)
        progress.update(task, advance=1)

        wait_for_healthy(state_service_container)
        progress.update(task, advance=1)

        wait_for_healthy(solution_tester_container)
        progress.update(task, advance=1)

        # Check Ollama API
        if not requests.get("http://localhost:11434").ok:
            raise Exception("Ollama API is not reachable at http://localhost:11434")

        CONSOLE.print("Ollama is healthy.", style=LOG_COLOR)
        progress.update(task, advance=1)

    return (
        ollama_container,
        web_ui_container,
        state_service_container,
        solution_tester_container,
        policy_models_container,
    )


def main(config: Config):
    global DOCKER_CLIENT
    global HF_TOKEN

    CONSOLE.print(
        """
                CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC                
            CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC            
         CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC         
       CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       
     CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC     
    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC    
  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  
  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  
 CCCCCCCCCCCCCC                                                  CCCCCCCCCCCCCC 
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
CCCCCCCCCCCCCCC                        CC                        CCCCCCCCCCCCCCC
 CCCCCCCCCCCCCC                                                  CCCCCCCCCCCCCC 
  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  
  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC  
    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC    
     CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC     
       CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC       
         CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC         
            CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC            
                CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC                
""",
        style=GENSYN_COLOR,
    )
    CONSOLE.print(
        "CodeAssist - Developed by Gensyn @ https://gensyn.ai/", style=GENSYN_COLOR
    )

    if not detect_docker():
        CONSOLE.print(
            "No container engine found. Please install and run Docker or Colima",
            style=ERROR_COLOR,
        )
        logger.error("Container engine not installed or not running.")
        return

    CONSOLE.print("Found a container engine", style=INFO_COLOR)

    if config.no_telemetry:
        logger.info("Telemetry is disabled.")
        CONSOLE.print("Telemetry is disabled.", style=WARNING_COLOR)

    if not HF_TOKEN:
        logger.warning("HuggingFace token not found")
        CONSOLE.print(
            "Please enter your HuggingFace token (you can get one from https://huggingface.co/settings/tokens):",
            style=INFO_COLOR,
        )
        CONSOLE.print("Note: Nothing will be shown as you type.", style=WARNING_COLOR)
        HF_TOKEN = CONSOLE.input("HuggingFace token: ", password=True).strip()
        if not HF_TOKEN:
            logger.error("HuggingFace token is required to proceed.")
            CONSOLE.print(
                "HuggingFace token is required to proceed. Exiting.", style=ERROR_COLOR
            )
            return

        with open(".env", "a") as f:
            f.write(f"\nHF_TOKEN={HF_TOKEN}\n")
        logger.info("HuggingFace token saved")

    logger.info("Starting CodeAssist...")
    setup_persistent_volume()

    network = ensure_network(config)

    CONSOLE.print(Markdown("## Setting up containers..."), style=INFO_COLOR)

    (
        ollama_container,
        web_ui_container,
        state_service_container,
        solution_tester_container,
        policy_models_container,
    ) = start_containers(config)

    logger.info("CodeAssist started")
    CONSOLE.print(Markdown("# CodeAssist Started"), style=HEADER_COLOR)

    if not config.train_only:
        browser.open("http://localhost:3000")

        CONSOLE.print(
            "A browser should have opened to http://localhost:3000. You can now start coding!",
            style=INFO_COLOR,
        )

    # Define containers dictionary once for reuse
    containers = {
        "ollama": ollama_container,
        "web_ui": web_ui_container,
        "state_service": state_service_container,
        "solution_tester": solution_tester_container,
        "policy_models": policy_models_container,
        "zero_style": None,
    }

    if not config.train_only:
        try:
            CONSOLE.print("Running... Press Ctrl+C trigger training", style=INFO_COLOR)
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received...")
            CONSOLE.print(
                Markdown("## Keyboard Interrupt Received"), style=HEADER_COLOR
            )

    if not config.no_train:
        # If training is enabled, stop only the web UI container
        CONSOLE.print(
            "Training enabled - stopping unneeded containers", style=INFO_COLOR
        )
        web_ui_container.stop()
        logger.info("Web UI container stopped.")
        CONSOLE.print("Web UI container stopped", style=LOG_COLOR)

        # Poll testing queue status every 10 seconds until it's empty
        CONSOLE.print("Waiting for testing queue to complete...", style=INFO_COLOR)
        await_testing_queue_completion()

        # Quick sweep for processed episode snapshots before training
        episodes_dir = PERSISTENT_DATA_DIR / "state-service" / "episodes"
        has_processed = False
        if episodes_dir.exists():
            for sub in episodes_dir.iterdir():
                if sub.is_dir():
                    snap = sub / f"{sub.name}.json"
                    if snap.is_file():
                        has_processed = True
                        break
        if not has_processed:
            CONSOLE.print(
                Markdown("## ⚠️ No processed episodes found"),
                style=WARNING_COLOR,
            )
            CONSOLE.print(
                f"No episode snapshots were found in {episodes_dir}. "
                "Training will be a no-op until you finish an episode and let tests run. "
                "Record at least one episode to completion (tests executed) and try again.",
                style=WARNING_COLOR,
            )
            CONSOLE.print("Skipping training.", style=WARNING_COLOR)
            stop_all_containers(containers)
            return

        zero_style_container = setup_zero_style_ui(config)
        containers["zero_style"] = zero_style_container

        CONSOLE.print(Markdown("## Running Policy Model Training"), style=HEADER_COLOR)
        training_success = run_training(config)
        if not training_success:
            CONSOLE.print(
                "Training failed. Continuing with other operations...",
                style=WARNING_COLOR,
            )
            logger.warning("Training failed. Continuing with other operations...")

    # If training is not enabled, stop all containers
    stop_all_containers(containers)

    cleanup_incomplete_episodes(PERSISTENT_DATA_DIR / "state-service" / "episodes")

    if not config.no_upload:
        # Upload to HuggingFace
        upload_to_huggingface(
            HF_TOKEN,
            folder_path=PERSISTENT_DATA_DIR / "trainer/models",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CodeAssist",
        description="AI coding assistant for software developers",
        epilog="Developed by Gensyn (https://gensyn.ai/)",
    )

    parser.add_argument(
        "-b", "--branch", help="Branch to use for all images", default="main"
    )
    parser.add_argument(
        "-o", "--ollama-tag", help="Ollama version tag", default="0.11.10"
    )
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry")
    parser.add_argument(
        "--no-pull", action="store_true", help="Do not pull Docker images"
    )
    parser.add_argument(
        "-n",
        "--network-name",
        help="Network name for Docker containers",
        default="codeassist_network",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Do not upload models to Hugging Face"
    )

    # Training arguments
    parser.add_argument(
        "--no-train", action="store_true", help="Disable policy model training"
    )
    parser.add_argument(
        "--training-config-path",
        help="Path to training configuration JSON file",
        default="training_config.json",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training and exit"
    )

    parser.add_argument(
        "--no-sc", action="store_true", help="Disable smart contract calls"
    )

    args = parser.parse_args()

    config = Config(**args.__dict__)

    main(config)
