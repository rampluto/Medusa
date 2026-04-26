# Hugging Face Hub / Space — deploy entry

This directory is the **Hub-facing deploy anchor** for the project.

## `Dockerfile` → one source of truth

[`Dockerfile`](Dockerfile) in this folder is a **symbolic link** to the
repository root [`Dockerfile`](../Dockerfile). There is a single
definition; you never copy-paste two copies out of sync.

- **Hugging Face Spaces** build the **root** `Dockerfile` on the branch
  you connect. You do not need to duplicate anything in `hf_space/` for
  the Space to work — the link is for clarity and for local builds.
- **Local Docker** from the repo root (context must be `.` so `COPY .`
  works):

  ```bash
  docker build -f hf_space/Dockerfile -t medusa-hf .
  # same image as: docker build -f Dockerfile -t medusa-hf .
  ```

If your OS or `git` does not preserve symlinks and `hf_space/Dockerfile`
arrives as a regular file, replace it with a copy of the root
`Dockerfile` or re-create: `ln -s ../Dockerfile hf_space/Dockerfile`.

## Push to Hugging Face

1. Push this **whole repository** to GitHub (or to the git remote the
   Space uses).
2. Space settings: **SDK = Docker**, **port 7860**, **GPU** enabled
   (T4+; set `MEDUSA_GRPO_LOAD_IN_4BIT=1` on T4 if you OOM).
3. Defaults are already in the root `Dockerfile`: `GRPO_MODEL_ID=anubhavkamal/medusa-qwen-grpo`, etc.
4. After build: `https://<user>-<space>.hf.space/medusa/studio` → agent **GRPO Trained** → **Auto-run**.
5. Add **Secret** `HF_TOKEN` only if downloads fail (gated `Qwen2.5-3B` or a private adapter). See [`trainer/README.md`](../trainer/README.md).

## Also in this tree

- GRPO predictor: [`trainer/grpo_predictor_hub.py`](../trainer/grpo_predictor_hub.py)
- Reference notes (SFT, GRPO, Space variables): [`trainer/README.md`](../trainer/README.md)
