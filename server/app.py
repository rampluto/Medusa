"""FastAPI server for the MEDUSA environment.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Via openenv CLI:
    openenv serve medusa_env
"""

from __future__ import annotations

import logging
import os

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


def _configure_logging() -> None:
    """Enable MEDUSA server logs with a predictable default format."""
    level_name = os.getenv("MEDUSA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger(__name__).info("logging_configured level=%s", logging.getLevelName(level))


_configure_logging()

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
