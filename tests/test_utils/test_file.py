from pathlib import Path, PurePath

import napari
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.utils.file import (
    open_parameter_file,
    serialize_registration_widget,
    write_parameter_file,
)


def test_open_parameter_file_with_valid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("(key1 value1 value2)\n(key2 value3 value4)")
    result = open_parameter_file(file_path)
    assert result == {
        "key1": ["value1", "value2"],
        "key2": ["value3", "value4"],
    }
    file_path.unlink()


def test_open_parameter_file_with_invalid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("invalid content")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


def test_open_parameter_file_with_empty_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


def test_open_parameter_file_with_comment_and_paren():
    """Test open_parameter_file handles entries with ) and / comments."""
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write(
            "(key1 value1 value2 )\n(key2 value3 / comment)\n(key3 value4)"
        )
    result = open_parameter_file(file_path)
    assert result == {
        "key1": ["value1", "value2"],
        "key2": ["value3"],
        "key3": ["value4"],
    }
    file_path.unlink()


def test_serialize_registration_widget(mocker):
    """Test serialize_registration_widget for different object types."""
    # Test with napari layer
    mock_layer = mocker.MagicMock(spec=napari.layers.Layer)
    mock_layer.name = "test_layer"
    assert serialize_registration_widget(mock_layer) == "test_layer"

    # Test with napari viewer
    mock_viewer = mocker.MagicMock(spec=napari.Viewer)
    mock_viewer.__str__ = mocker.MagicMock(return_value="test_viewer")
    assert serialize_registration_widget(mock_viewer) == "test_viewer"

    # Test with PurePath
    test_path = PurePath("test/path")
    assert serialize_registration_widget(test_path) == str(test_path)

    # Test with BrainGlobeAtlas
    mock_atlas = mocker.MagicMock(spec=BrainGlobeAtlas)
    mock_atlas.atlas_name = "test_atlas"
    assert serialize_registration_widget(mock_atlas) == "test_atlas"

    # Test with numpy array
    test_array = np.array([[1, 2], [3, 4]])
    result = serialize_registration_widget(test_array)
    assert "<class 'numpy.ndarray'>" in result
    assert "(2, 2)" in result
    assert test_array.dtype.name in result or str(test_array.dtype) in result

    # Test with other object that has __dict__() method
    # (like RegistrationWidget)
    class TestObj:
        def __dict__(self):
            return {"key": "value"}

    test_obj = TestObj()
    assert serialize_registration_widget(test_obj) == {"key": "value"}


def test_write_parameter_file_round_trip(tmp_path: Path):
    param_dict = {
        "Interpolator": ["LinearInterpolator"],
        "NumberOfHistogramBins": ["32"],
        "MixedValues": ["foo", "1.25"],
    }
    file_path = tmp_path / "params.txt"

    write_parameter_file(file_path, param_dict)

    loaded = open_parameter_file(file_path)
    assert loaded == param_dict


def test_write_parameter_file_quotes_strings(tmp_path: Path):
    param_dict = {"Transform": ["AffineTransform"], "DefaultPixelValue": ["0"]}
    file_path = tmp_path / "params.txt"

    write_parameter_file(file_path, param_dict)

    contents = file_path.read_text()
    assert '(Transform "AffineTransform")' in contents
    assert "(DefaultPixelValue 0)" in contents
