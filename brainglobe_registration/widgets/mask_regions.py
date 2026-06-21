from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AtlasRegionMaskWidget(QWidget):
    """
    A collapsible widget that displays a hierarchical
    tree of atlas regions.

    Users can check individual regions (and their descendants)
    to exclude them from registration via a binary mask.
    selected_region_ids is the set of directly checked region IDs;
    descendants are collected when building the actual mask.

    Signals
    -------
    region_toggled : Signal(int, str, bool)
        Emitted whenever a region is (un)checked by the user.
        Carries (region_id, region_name, checked).
    """

    region_toggled = Signal(int, str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._selected_region_ids: set[int] = set()
        self._children_lookup: dict[int, list[int]] = {}
        self._item_by_id: dict[int, QTreeWidgetItem] = {}
        self._propagating = False

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # toggle button (acts as the "dropdown" header)
        self._toggle_button = QPushButton("▶  Mask atlas regions (optional)")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._toggle_button.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; }"
        )
        self._toggle_button.toggled.connect(self._on_toggle)
        outer_layout.addWidget(self._toggle_button)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setVerticalScrollMode(QTreeWidget.ScrollPerPixel)
        self.tree.setMinimumHeight(180)

        self.tree.itemChanged.connect(self._on_item_changed)

        outer_layout.addWidget(self.tree)

        self.tree.setVisible(False)
        self.setLayout(outer_layout)

    @property
    def selected_region_ids(self) -> set[int]:
        """
        The set of region IDs that are directly checked by the user.
        Descendant IDs are not included here.
        """
        return self._selected_region_ids

    def populate_from_atlas(self, atlas) -> None:
        self.tree.itemChanged.disconnect(self._on_item_changed)
        self.tree.clear()
        self._selected_region_ids.clear()
        self._item_by_id.clear()

        structures = atlas.structures

        children_lookup: dict[int | None, list[int]] = {}
        for sid, structure in structures.items():
            path = structure.get("structure_id_path") or [sid]
            # path is [root_id, ..., parent_id, sid]; second-last is
            # the direct parent, or None if this is itself the root.
            parent_id = path[-2] if len(path) > 1 else None
            children_lookup.setdefault(parent_id, []).append(sid)

        self._children_lookup = {
            k: v for k, v in children_lookup.items() if k is not None
        }

        for root_id in children_lookup.get(None, []):
            self._add_structure_recursive(
                parent_item=None,
                structure_id=root_id,
                structures=structures,
                children_lookup=children_lookup,
            )

        self.tree.expandToDepth(1)
        self.tree.itemChanged.connect(self._on_item_changed)

    def get_descendant_ids(self, region_id: int) -> set[int]:
        """
        Return `region_id` together with every id in its subtree.
        """
        ids = {region_id}
        for child_id in self._children_lookup.get(region_id, []):
            ids |= self.get_descendant_ids(child_id)
        return ids

    def uncheck_region(self, region_id: int) -> None:
        """
        Programmatically uncheck a region (e.g. if its mask layer was
        deleted directly from the napari viewer) without recursing.
        """
        item = self._item_by_id.get(region_id)
        if item is not None:
            item.setCheckState(0, Qt.Unchecked)

    def _on_toggle(self, checked: bool) -> None:
        self.tree.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self._toggle_button.setText(f"{arrow}  Mask atlas regions (optional)")

    def _add_structure_recursive(
        self,
        parent_item: QTreeWidgetItem | None,
        structure_id: int,
        structures: dict,
        children_lookup: dict,
    ) -> None:
        structure = structures[structure_id]

        item = QTreeWidgetItem()
        item.setText(0, structure["name"])
        item.setData(0, Qt.UserRole, structure_id)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Unchecked)
        self._item_by_id[structure_id] = item

        if parent_item is None:
            self.tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)

        for child_id in children_lookup.get(structure_id, []):
            self._add_structure_recursive(
                parent_item=item,
                structure_id=child_id,
                structures=structures,
                children_lookup=children_lookup,
            )

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._propagating:
            return

        region_id: int = item.data(0, Qt.UserRole)
        checked = item.checkState(0) == Qt.Checked

        if checked:
            self._selected_region_ids.add(region_id)
        else:
            self._selected_region_ids.discard(region_id)

        self._propagating = True
        try:
            self._set_subtree_checkstate(item, item.checkState(0))
        finally:
            self._propagating = False

        self.region_toggled.emit(region_id, item.text(0), checked)

    def _set_subtree_checkstate(
        self, item: QTreeWidgetItem, state: Qt.CheckState
    ) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, state)
            self._set_subtree_checkstate(child, state)
