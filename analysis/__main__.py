"""允許透過 `python -m analysis` 執行。"""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover - 直接透過 CLI 執行
    main()
