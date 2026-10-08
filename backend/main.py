"""VestiAI entry point.

Usage::

    python -m backend.main                      # start the web app on :8000
    python -m backend.main --port 9000 --reload
    python run.py --host 0.0.0.0 --port 8000    # equivalent root-level shortcut
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python backend/main.py` as well as `python -m backend.main`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.app import create_app  # noqa: E402
from backend.config import load_settings  # noqa: E402
from backend.utils.logging_utils import get_logger, setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VestiAI web application.")
    parser.add_argument("--host", default=None, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port (default: 8000)")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (keep at 1: the model is stateful)")
    parser.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--force-cpu", action="store_true", help="Ignore CUDA/MPS and run on CPU")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = int(args.port)
    if args.log_level:
        settings.log_level = args.log_level
    if args.force_cpu:
        settings.force_cpu = True

    setup_logging(settings.log_level, settings.log_dir)
    logger = get_logger("main")

    # Respect container/platform hints: bind 0.0.0.0 so the sandbox preview proxy works.
    host = os.environ.get("VESTIAI_HOST", settings.host)
    port = int(os.environ.get("VESTIAI_PORT", os.environ.get("PORT", settings.port)))

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        logger.error("uvicorn is required to run the server: pip install uvicorn[standard] (%s)", exc)
        return 1

    logger.info("Starting %s %s on http://%s:%s", settings.app_name, settings.app_version, host, port)
    logger.info("Open the app at http://localhost:%s  ·  API docs at /docs  ·  status at /api/status", port)

    if args.workers > 1:
        logger.warning(
            "Multiple workers share no model state and will each load the model into VRAM. "
            "Keep --workers 1 unless you provide an external model server."
        )

    uvicorn.run(
        "backend.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=max(1, args.workers),
        log_level=str(settings.log_level).lower(),
        access_log=False if not settings.debug else True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
