"""FastAPI server for the MEDUSA environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Via openenv CLI:
    openenv serve medusa_env
"""

from __future__ import annotations

# Support three import contexts:
#   1. In-repo (from OpenEnv root): relative imports via `..`
#   2. Standalone installed (uv run server): medusa_env.* package
#   3. Direct execution inside env dir: bare module names
try:
    from openenv.core.env_server.http_server import create_app
    from ..medusa_env import MedusaEnv
    from ..models import MedusaAction, MedusaObservation
except ImportError:
    try:
        from openenv.core.env_server.http_server import create_app
        from medusa_env import MedusaEnv
        from medusa_env.models import MedusaAction, MedusaObservation
    except ImportError:
        from openenv.core.env_server.http_server import create_app
        from medusa_env import MedusaEnv  # type: ignore[no-redef]
        from models import MedusaAction, MedusaObservation  # type: ignore[no-redef]

app = create_app(
    MedusaEnv,
    MedusaAction,
    MedusaObservation,
    env_name="medusa_env",
)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
