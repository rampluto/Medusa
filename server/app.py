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
from openenv.core.env_server.http_server import create_app
try:
    from ..models import MedusaAction, MedusaObservation
    from .custom_api import register_custom_routes
    from .medusa_env import MedusaEnv
except ImportError:
    try:
        from medusa_env.models import MedusaAction, MedusaObservation
        from medusa_env.server.custom_api import register_custom_routes
        from medusa_env.server.medusa_env import MedusaEnv
    except ImportError:
        from models import MedusaAction, MedusaObservation
        from server.custom_api import register_custom_routes
        from server.medusa_env import MedusaEnv

app = create_app(
    MedusaEnv,
    MedusaAction,
    MedusaObservation,
    env_name="medusa_env",
)
register_custom_routes(app)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
