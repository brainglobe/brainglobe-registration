import pytest
from qtpy.QtCore import Qt

from brainglobe_registration.widgets.mask_regions import MaskRegionsDialog


class FakeAtlas:
    """
    Minimal BrainGlobeAtlas with only the .structures mapping that
    populate_from_atlas reads.
    """

    def __init__(self, structures: dict):
        self.structures = structures


@pytest.fixture()
def fake_atlas_structures():
    # Create a small tree:
    # 1 (root)
    # |- 2 (Region A)
    # |  |- 4 (Region A1)
    # |  |- 5 (Region A2)
    # |- 3 (Region B)
    return {
        1: {"name": "root", "structure_id_path": [1]},
        2: {"name": "Region A", "structure_id_path": [1, 2]},
        3: {"name": "Region B", "structure_id_path": [1, 3]},
        4: {"name": "Region A1", "structure_id_path": [1, 2, 4]},
        5: {"name": "Region A2", "structure_id_path": [1, 2, 5]},
    }


@pytest.fixture()
def dialog(qtbot, fake_atlas_structures):
    dlg = MaskRegionsDialog()
    qtbot.addWidget(dlg)
    dlg.populate_from_atlas(FakeAtlas(fake_atlas_structures))
    return dlg


def test_init_no_atlas(qtbot):
    dlg = MaskRegionsDialog()
    qtbot.addWidget(dlg)

    assert dlg.selected_region_ids == set()
    assert dlg.tree.topLevelItemCount() == 0


def test_populate_from_atlas_builds_tree(dialog):
    assert dialog.tree.topLevelItemCount() == 1

    root_item = dialog.tree.topLevelItem(0)
    assert root_item.data(0, Qt.UserRole) == 1
    assert root_item.childCount() == 2

    child_ids = {
        root_item.child(i).data(0, Qt.UserRole)
        for i in range(root_item.childCount())
    }
    assert child_ids == {2, 3}

    region_a_item = dialog._item_by_id[2]
    assert region_a_item.childCount() == 2
    grandchild_ids = {
        region_a_item.child(i).data(0, Qt.UserRole)
        for i in range(region_a_item.childCount())
    }
    assert grandchild_ids == {4, 5}


def test_populate_from_atlas_resets_previous_state(
    dialog, fake_atlas_structures
):
    dialog._item_by_id[2].setCheckState(0, Qt.Checked)
    assert dialog.selected_region_ids == {2}

    dialog.populate_from_atlas(FakeAtlas(fake_atlas_structures))
    assert dialog.selected_region_ids == set()


def test_get_descendant_ids(dialog):
    assert dialog.get_descendant_ids(2) == {2, 4, 5}
    assert dialog.get_descendant_ids(3) == {3}
    assert dialog.get_descendant_ids(1) == {1, 2, 3, 4, 5}


def test_checking_region_adds_only_itself_to_selected_ids(dialog):
    dialog._item_by_id[2].setCheckState(0, Qt.Checked)

    # Only the directly-checked id is tracked; descendants are resolved
    # separately via get_descendant_ids when building the mask.
    assert dialog.selected_region_ids == {2}


def test_checking_region_and_descendants(dialog):
    dialog._item_by_id[2].setCheckState(0, Qt.Checked)

    assert dialog._item_by_id[4].checkState(0) == Qt.Checked
    assert dialog._item_by_id[5].checkState(0) == Qt.Checked
    # Sibling subtree is untouched
    assert dialog._item_by_id[3].checkState(0) == Qt.Unchecked


def test_unchecking_region_removes_from_selected_ids(dialog):
    dialog._item_by_id[2].setCheckState(0, Qt.Checked)
    dialog._item_by_id[2].setCheckState(0, Qt.Unchecked)

    assert dialog.selected_region_ids == set()
    assert dialog._item_by_id[4].checkState(0) == Qt.Unchecked
    assert dialog._item_by_id[5].checkState(0) == Qt.Unchecked


def test_region_toggled_signal_emitted_on_check(qtbot, dialog):
    with qtbot.waitSignal(dialog.region_toggled, timeout=1000) as blocker:
        dialog._item_by_id[3].setCheckState(0, Qt.Checked)

    assert blocker.args == [3, "Region B", True]


def test_region_toggled_signal_emitted_on_uncheck(qtbot, dialog):
    dialog._item_by_id[3].setCheckState(0, Qt.Checked)

    with qtbot.waitSignal(dialog.region_toggled, timeout=1000) as blocker:
        dialog._item_by_id[3].setCheckState(0, Qt.Unchecked)

    assert blocker.args == [3, "Region B", False]


def test_uncheck_region_updates_item_and_selection(dialog):
    dialog._item_by_id[2].setCheckState(0, Qt.Checked)
    assert dialog.selected_region_ids == {2}

    dialog.uncheck_region(2)

    assert dialog._item_by_id[2].checkState(0) == Qt.Unchecked
    assert dialog.selected_region_ids == set()


def test_done_button_accepts_dialog(qtbot, dialog):
    with qtbot.waitSignal(dialog.accepted, timeout=1000):
        dialog.close_button.click()
