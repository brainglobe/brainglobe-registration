import napari
import pytest
from PIL import Image
import numpy as np

from brainglobe_registration.registration_widget import RegistrationWidget


@pytest.fixture()
def make_napari_viewer_with_images(make_napari_viewer, pytestconfig):
    viewer: napari.Viewer = make_napari_viewer()

    root_path = pytestconfig.rootpath
    atlas_image = Image.open(
        root_path / "src/tests/test_images/Atlas_Hipp.tif"
    )
    moving_image = Image.open(
        root_path / "src/tests/test_images/sample_hipp.tif"
    )

    viewer.add_image(np.asarray(moving_image), name="moving_image")
    viewer.add_image(np.asarray(atlas_image), name="atlas_image")

    return viewer


@pytest.fixture()
def registration_widget(make_napari_viewer_with_images):
    viewer = make_napari_viewer_with_images

    widget = RegistrationWidget(viewer)

    return widget


def test_registration_widget(make_napari_viewer_with_images):
    widget = RegistrationWidget(make_napari_viewer_with_images)

    assert widget is not None


# @pytest.mark.parametrize(
#     "name, index",
#     [
#         ("moving_image", 0),
#         ("atlas_image", 1),
#     ],
# )
# def test_find_layer_index(make_napari_viewer_with_images, name, index):
#     viewer = make_napari_viewer_with_images
#
#     widget = RegistrationWidget(viewer)
#
#     assert widget.find_layer_index(name) == index
#
#
# def test_get_image_layer_names(make_napari_viewer_with_images):
#     viewer = make_napari_viewer_with_images
#
#     widget = RegistrationWidget(viewer)
#     layer_names = widget.get_image_layer_names()
#
#     assert len(layer_names) == 2
#     assert layer_names == ["moving_image", "atlas_image"]


def test_atlas_dropdown_index_changed_with_valid_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(2)

    assert (
        registration_widget._atlas.atlas_name
        == registration_widget._available_atlases[2]
    )
    assert registration_widget.run_button.isEnabled()
    assert registration_widget._viewer.grid.enabled


def test_atlas_dropdown_index_changed_with_zero_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(0)

    assert registration_widget._atlas is None
    assert not registration_widget.run_button.isEnabled()
    assert not registration_widget._viewer.grid.enabled


def test_atlas_dropdown_index_changed_with_existing_atlas(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(2)

    registration_widget._on_atlas_dropdown_index_changed(1)

    assert (
        registration_widget._atlas.atlas_name
        == registration_widget._available_atlases[1]
    )
    assert registration_widget.run_button.isEnabled()
    assert registration_widget._viewer.grid.enabled


def test_sample_dropdown_index_changed_with_valid_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_sample_dropdown_index_changed(1)

    assert (
        registration_widget._moving_image.name
        == registration_widget._sample_images[1]
    )
