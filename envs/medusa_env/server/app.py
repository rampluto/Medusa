"""FastAPI server for the MEDUSA environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Via openenv CLI:
    openenv serve medusa_env
"""

from __future__ import annotations

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app

    from ..medusa_env import MedusaEnv
    from ..models import MedusaAction, MedusaObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app

    from medusa_env import MedusaEnv
    from models import MedusaAction, MedusaObservation

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
