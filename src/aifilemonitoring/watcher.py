from __future__ import annotations

import logging
import time
from pathlib import Path

from .models import PipelineConfig
from .processor import FileProcessor

LOGGER = logging.getLogger(__name__)


class PollingFileWatcher:
    """Cross-platform polling watcher that runs continuously."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processor = FileProcessor(config)
        self._seen_sizes: dict[Path, tuple[int, float]] = {}

    def run_forever(self) -> None:
        self.config.ensure_directories()
        LOGGER.info("Watching %s for %s", self.config.watch_dir, self.config.input_glob)
        while True:
            self.scan_once()
            time.sleep(self.config.poll_seconds)

    def scan_once(self) -> list[dict[str, int | str]]:
        processed = []
        for candidate in sorted(self.config.watch_dir.glob(self.config.input_glob)):
            if not candidate.is_file() or candidate.parent != self.config.watch_dir:
                continue
            if self._is_stable(candidate):
                LOGGER.info("Processing stable file %s", candidate)
                processed.append(self.processor.process(candidate))
                self._seen_sizes.pop(candidate, None)
        return processed

    def _is_stable(self, path: Path) -> bool:
        stat = path.stat()
        size = stat.st_size
        now = time.monotonic()
        previous = self._seen_sizes.get(path)
        if not previous or previous[0] != size:
            self._seen_sizes[path] = (size, now)
            return False
        return now - previous[1] >= self.config.stable_seconds
