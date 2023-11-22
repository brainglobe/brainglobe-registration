import napari
import pytest
from PIL import Image
import numpy as np

from brainglobe_registration.registration_widget import RegistrationWidget
from bg_atlasapi import BrainGlobeAtlas


atlas_image = Image.open("src/tests/test_images/Atlas_Hipp.tif")
moving_image = Image.open("src/tests/test_images/sample_hipp.tif")


@pytest.fixture()
def make_napari_viewer_with_images(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer()

    viewer.add_image(np.asarray(moving_image), name="moving_image")
    viewer.add_image(np.asarray(atlas_image), name="atlas_image")

    return viewer


def test_registration_widget(make_napari_viewer_with_images):
    widget = RegistrationWidget(make_napari_viewer_with_images)

    assert widget is not None


def test_registration_widget_atlas_selection_initial(
    make_napari_viewer_with_images,
):
    widget = RegistrationWidget(make_napari_viewer_with_images)
    widget.available_atlas_dropdown.setCurrentIndex(1)

    assert widget._atlas is not None
    assert widget._atlas.atlas_name == "allen_mouse_100um"


def test_registration_widget_atlas_already_added(
    make_napari_viewer_with_images,
):
    widget = RegistrationWidget(make_napari_viewer_with_images)
    widget._atlas = BrainGlobeAtlas("allen_mouse_100um")

    widget.available_atlas_dropdown.setCurrentIndex(1)

    assert widget._atlas is not None
    assert widget._atlas.atlas_name == "allen_mouse_100um"


def test_registration_widget_atlas_switch_atlas(
    make_napari_viewer_with_images,
):
    widget = RegistrationWidget(make_napari_viewer_with_images)
    widget._atlas = BrainGlobeAtlas("allen_mouse_100um")

    widget.available_atlas_dropdown.setCurrentIndex(2)

    assert widget._atlas is not None
    assert widget._atlas.atlas_name == "example_mouse_100um"


@pytest.mark.parametrize(
    "name, index",
    [
        ("moving_image", 0),
        ("atlas_image", 1),
    ],
)
def test_find_layer_index(make_napari_viewer_with_images, name, index):
    viewer = make_napari_viewer_with_images

    widget = RegistrationWidget(viewer)

    assert widget.find_layer_index(name) == index


def test_get_image_layer_names(make_napari_viewer_with_images):
    viewer = make_napari_viewer_with_images

    widget = RegistrationWidget(viewer)
    layer_names = widget.get_image_layer_names()

    assert len(layer_names) == 2
    assert layer_names == ["moving_image", "atlas_image"]
