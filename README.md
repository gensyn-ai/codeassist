
# CodeAssist - AI Programming Assistant

CodeAssist is a completely private and local AI coding assistant, developed by Gensyn. It helps you practice programming problems and train a novel assistant to help you code.

Unlike typical code assistants, CodeAssist writes directly in your editor as you work. Every keystroke - whether you type, fix, delete, or leave its output untouched - becomes a learning signal. Over time, it adapts to your habits and style, acting more like an apprentice learning from your craft than a tool following commands.

[Docs](https://docs.gensyn.ai/testnet/codeassist) | [Tutorial](https://docs.gensyn.ai/testnet/codeassist/using-codeassist) | [Leaderboard](https://dashboard.gensyn.ai/?application=CodeAssist)

# Installation

Get started with installing CodeAssist.

## Docker

Install [Docker](https://docs.docker.com/engine/install/) on your system, according to the instructions for your machine.

## Python

Python is required to run the main script that handles your environment. We require a version no older than 3.10.

## UV

UV is required to manage the dependencies of the main script. It can be installed with the following steps:

### MacOS

```bash
brew install uv
```

### Linux (or alternate MacOS install, for those without Brew)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Downloading the Code

To download the code, simply clone the repository:

```bash
git clone https://github.com/gensyn-ai/codeassist.git
cd codeassist
```

# Running

To run CodeAssist, simply execute the following command:

```bash
uv run run.py
```

## HuggingFace Token

To start CodeAssist, you will need to have a HuggingFace token. Follow [these instructions](https://huggingface.co/docs/hub/en/security-tokens) and generate a token with `Write` access.

## Web UI

After the script is running, your browser should open automatically but if it doesn't, open a window and go to [localhost:3000](http://localhost:3000) to open CodeAssist.

When the web UI loads, you'll see a login modal where you can log in with email (which sends a one-time passcode) or with Google. After logging in for the first time, your local credentials will be stored in `persistent-data/auth/userKeyMap.json`.

Once logged in, you can select Easy, Medium, or Hard problems from the sidebar. CodeAssist will begin recording an episode. Every click, keystroke, edit, or deletion is logged as training feedback.

When you stop typing, CodeAssist takes initiative. It writes directly into your file without any pop-ups or confirmations. Whether you accept, modify, or remove its edits, each interaction contributes to the model's understanding of your preferences.

### Tips & Tricks

- Use `Shift+Space` or click the Pause Assistant button to temporarily stop the assistant. The first keystroke after pausing will unpause it.
- Keep your cursor near the section you're working on, as CodeAssist inserts code relative to your cursor position.
- When the assistant produces a "No-Op" (does nothing), it's waiting for you. This is intentional and signals it's your turn to act.

## Training

CodeAssist continuously records your interactions while the web UI is running. To complete an episode and train your model, press `Ctrl+C` in the terminal where CodeAssist is running.

You do not need to successfully solve a LeetCode problem to train the model. You can stop recording the episode by leaving the CodeAssist web UI, returning to the terminal CodeAssist is running in, and using the `ctrl+c` command to start training.

During training, CodeAssist will:
- Compare your edits to the assistant's actions
- Calculate rewards and penalties based on your interactions
- Update your local model checkpoint
- Store new model weights under `persistent-data/trainer/models`
- Upload your trained model to Hugging Face (if a valid token is provided)

After training completes (which takes a few minutes depending on your system), you can restart CodeAssist to use your updated model trained on your most recent episode.

### Best Practices

- **Be patient**: The assistant watches your typing and timing. Working too quickly or aggressively correcting can neutralize training efficacy.
- **Treat it as a collaborator**: Let it naturally interject code and keep useful code around briefly before editing or removing.
- **Don't delete everything instantly**: If you delete everything it writes right away, you're teaching it to stop acting altogether.
- **Record multiple varied problems**: Diversify its learning signals by working on different problems.
- **Expect gradual improvement**: Early episodes may feel inconsistent. Improvement becomes clearer after 4-5 episodes of training.

# Troubleshooting

## Exception: Container <container-name> is unhealthy.

This occurs when a container fails to boot. You can view the logs by running `docker logs <container-name>`. Please review and upload the logs when creating a new issue.

## Error connecting to Docker daemon

```
2025-09-04 15:03:47,975 - ERROR - Error connecting to Docker daemon: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))
2025-09-04 15:03:47,976 - ERROR - Please ensure Docker is installed and running.
```

If you see lines like the above, it means Docker is not running, or you do not have permission to connect to Docker. Make sure Docker is properly installed on your system and running.

## Bind for 0.0.0.0:3000 failed: port is already allocated

This occurs when there is already a Docker container running that uses port 3000 or other required ports.

### Port Configuration

CodeAssist uses multiple services, each with configurable ports. If you encounter port conflicts, you can customize any of the following:

```bash
uv run run.py --port 3001 --ollama-port 11435 --state-service-port 8002 --solution-tester-port 8009 --policy-model-port 8003
```

#### Available Port Options:

- `--port` : Web UI port (default: 3000)
- `--ollama-port` : Ollama API port (default: 11434)
- `--state-service-port` : State Service port (default: 8000)
- `--solution-tester-port` : Solution Tester port (default: 8008)
- `--policy-model-port` : Policy Model API port (default: 8001)

#### Examples:

**Change only the Web UI port:**
```bash
uv run run.py --port 3001
```

**Change multiple ports to avoid conflicts:**
```bash
uv run run.py --port 3001 --ollama-port 11435
```

> **Note:** Zero-style UI port is fixed at 3003. When choosing custom ports, avoid `8080` and `3003`, as they may conflict with other services.

# Contributing

We welcome contributions!

## Pre-Commit Hook

We use a pre-commit hook to run linting on the repository before committing. You can install `pre-commit` as described below:

### Installing `pre-commit`

To install `pre-commit`, simply run the below command

```bash
pip install pre-commit
```

### Setting up `pre-commit`

To set up `pre-commit`, run this command

```bash
pre-commit install
```

## Formatting

This repository is formatted with `ruff format`. Any commits which do not meet Ruff's lint checks will have GitHub actions fail, and will need to be fixed before merging.

# License

CodeAssist is released under the (MIT) license.
