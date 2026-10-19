import sys
from pathlib import Path
from types import TracebackType


class ImportPath:
    """Context manager that puts ``path`` first on ``sys.path`` inside the block."""

    def __init__(self, path: Path) -> None:
        self._entry = str(path)
        self._added = False

    def __enter__(self) -> None:
        if self._entry not in sys.path:
            sys.path.insert(0, self._entry)
            self._added = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._added:
            sys.path.remove(self._entry)
            self._added = False
