from __future__ import annotations

import pathlib


def main() -> None:
    models_dir = pathlib.Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    print("TODO: implement secure model download and checksum verification")


if __name__ == "__main__":
    main()
