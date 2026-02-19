import pytest

from brainglobe_registration.widgets.select_images_view import SelectImagesView

available_atlases = [
    "------",
    "allen_mouse_100um",
    "allen_mouse_25um",
    "example_mouse_100um",
]
sample_image_names = ["image1", "image2", "image3"]


@pytest.fixture(scope="class")
def select_images_view() -> SelectImagesView:
    select_images_view = SelectImagesView(
        available_atlases=available_atlases,
        sample_image_names=sample_image_names,
    )
    return select_images_view


def test_select_images_view(select_images_view, qtbot):
    qtbot.addWidget(select_images_view)

    assert select_images_view.available_atlas_dropdown.count() == len(
        available_atlases
    )
    assert select_images_view.available_sample_dropdown.count() == len(
        sample_image_names
    )
    assert (
        select_images_view.available_atlas_dropdown.currentText()
        == available_atlases[0]
    )
    assert (
        select_images_view.available_sample_dropdown.currentText()
        == sample_image_names[0]
    )
    assert (
        select_images_view.available_atlas_dropdown_label.text()
        == "Select Atlas:"
    )
    assert (
        select_images_view.available_sample_dropdown_label.text()
        == "Select sample:"
    )
    # Test sample geometry dropdown
    assert (
        select_images_view.sample_geometry_label.text() == "Sample Geometry:"
    )
    assert select_images_view.sample_geometry_dropdown.count() == 3
    assert (
        select_images_view.sample_geometry_dropdown.currentText()
        == "Full Brain"
    )


@pytest.mark.parametrize(
    "atlas_index, expected",
    [
        (1, 1),
        (2, 2),
        (len(available_atlases), -1),
    ],
)
def test_select_images_view_atlas_index_change_once(
    select_images_view, qtbot, atlas_index, expected
):
    qtbot.addWidget(select_images_view)

    with qtbot.waitSignal(
        select_images_view.atlas_index_change, timeout=1000
    ) as blocker:
        select_images_view.available_atlas_dropdown.setCurrentIndex(
            atlas_index
        )
    assert blocker.args == [expected]


@pytest.mark.parametrize(
    "atlas_indexes",
    [
        ([1, 2, 0]),
        ([1, 2, 1]),
    ],
)
def test_select_images_view_atlas_index_change_multi(
    select_images_view, qtbot, atlas_indexes
):
    qtbot.addWidget(select_images_view)

    expected = -1

    def check_index_valid(signal_index):
        return signal_index == expected

    with qtbot.waitSignals(
        [select_images_view.atlas_index_change] * 3,
        check_params_cbs=[check_index_valid] * 3,
        timeout=1000,
    ):
        for index in atlas_indexes:
            expected = index
            select_images_view.available_atlas_dropdown.setCurrentIndex(index)


@pytest.mark.parametrize(
    "image_index, expected",
    [
        (1, 1),
        (2, 2),
        (len(sample_image_names), -1),
    ],
)
def test_select_images_view_moving_image_index_change_once(
    select_images_view, qtbot, image_index, expected
):
    qtbot.addWidget(select_images_view)

    with qtbot.waitSignal(
        select_images_view.moving_image_index_change, timeout=1000
    ) as blocker:
        select_images_view.available_sample_dropdown.setCurrentIndex(
            image_index
        )
    assert blocker.args == [expected]


@pytest.mark.parametrize(
    "image_indexes",
    [
        ([1, 2, 0]),
        ([1, 2, 1]),
    ],
)
def test_select_images_view_moving_image_index_change_multi(
    select_images_view, qtbot, image_indexes
):
    qtbot.addWidget(select_images_view)

    expected = -1

    def check_index_valid(signal_index):
        return signal_index == expected

    with qtbot.waitSignals(
        [select_images_view.atlas_index_change] * 3,
        check_params_cbs=[check_index_valid] * 3,
        timeout=1000,
    ):
        for index in image_indexes:
            expected = index
            select_images_view.available_atlas_dropdown.setCurrentIndex(index)


@pytest.mark.parametrize(
    "geometry_index, expected_geometry",
    [
        (0, "full"),
        (1, "hemisphere_l"),
        (2, "hemisphere_r"),
    ],
)
def test_select_images_view_sample_geometry_change(
    select_images_view, qtbot, geometry_index, expected_geometry
):
    qtbot.addWidget(select_images_view)

    # For index 0, we need to set it to a different value first,
    # because Qt doesn't emit currentIndexChanged when setting to the current index
    if geometry_index == 0:
        select_images_view.sample_geometry_dropdown.setCurrentIndex(1)

    with qtbot.waitSignal(
        select_images_view.sample_geometry_change, timeout=1000
    ) as blocker:
        select_images_view.sample_geometry_dropdown.setCurrentIndex(
            geometry_index
        )
    assert blocker.args == [expected_geometry]


def test_select_images_view_default_geometry(select_images_view, qtbot):
    qtbot.addWidget(select_images_view)

    # Default should be "full" (index 0)
    assert select_images_view.sample_geometry_dropdown.currentIndex() == 0
    assert (
        select_images_view.sample_geometry_dropdown.currentText()
        == "Full Brain"
    )
