"""Run the PixelPipes unit-test suite with the repository's test settings."""

from pathlib import Path
import unittest


def main() -> int:
    tests_dir = Path(__file__).parent
    repository_root = tests_dir.parent
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir),
        pattern="test_*.py",
        top_level_dir=str(repository_root),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
