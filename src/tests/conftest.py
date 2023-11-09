import os
from pathlib import Path

import pytest
from bg_atlasapi import BrainGlobeAtlas, config as bg_config


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        dest="slow",
        default=False,
        help="enable runslow decorated tests",
    )


def pytest_configure(config):
    if not config.option.slow:
        setattr(config.option, "markexpr", "not slow")


@pytest.fixture(autouse=True)
def mock_brainglobe_user_folders(monkeypatch):
    """Ensures user config and data is mocked during all local testing.

    User config and data need mocking to avoid interfering with user data.
    Mocking is achieved by turning user data folders used in tests into
    subfolders of a new ~/.brainglobe-tests folder instead of ~/.

    It is not sufficient to mock the home path in the tests, as this
    will leave later imports in other modules unaffected.

    GH actions workflow will test with default user folders.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        home_path = Path.home()  # actual home path
        mock_home_path = home_path / ".brainglobe-tests"
        if not mock_home_path.exists():
            mock_home_path.mkdir()

        def mock_home():
            return mock_home_path

        monkeypatch.setattr(Path, "home", mock_home)

        # also mock global variables of config.py
        monkeypatch.setattr(
            bg_config, "DEFAULT_PATH", mock_home_path / ".brainglobe"
        )
        monkeypatch.setattr(
            bg_config, "CONFIG_DIR", mock_home_path / ".config" / "brainglobe"
        )
        monkeypatch.setattr(
            bg_config,
            "CONFIG_PATH",
            bg_config.CONFIG_DIR / bg_config.CONFIG_FILENAME,
        )
        mock_default_dirs = {
            "default_dirs": {
                "brainglobe_dir": mock_home_path / ".brainglobe",
                "interm_download_dir": mock_home_path / ".brainglobe",
            }
        }
        monkeypatch.setattr(bg_config, "TEMPLATE_CONF_DICT", mock_default_dirs)


@pytest.fixture(autouse=True)
def setup_preexisting_local_atlases():
    """Automatically setup all tests to have three downloaded atlases
    in the test user data."""
    preexisting_atlases = [
        ("example_mouse_100um", "v1.2"),
        ("allen_mouse_100um", "v1.2"),
        ("osten_mouse_100um", "v1.1"),
    ]
    for atlas_name, version in preexisting_atlases:
        if not Path.exists(
            Path.home() / f".brainglobe/{atlas_name}_{version}"
        ):
            _ = BrainGlobeAtlas(atlas_name)
