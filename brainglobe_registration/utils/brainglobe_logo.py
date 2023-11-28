from importlib.resources import files

from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

brainglobe_logo = files("brainglobe_registration").joinpath(
    "resources/brainglobe.png"
)

_logo_html = f"""
<h1>
<img src="{brainglobe_logo}"width="100">
<\h1>
"""


def _docs_links_widget(parent: QWidget = None):
    _docs_links_html = """
    <h3>
    <p><a href="https://brainglobe.info" style="color:gray;">Website</a></p>
    <p><a href="https://github.com/brainglobe/brainglobe-registration" style="color:gray;">Source</a></p>
    </h3>
    """  # noqa: E501
    docs_links_widget = QLabel(_docs_links_html, parent=parent)
    docs_links_widget.setOpenExternalLinks(True)
    return docs_links_widget


def _logo_widget(parent: QWidget = None):
    return QLabel(_logo_html, parent=None)


def header_widget(parent: QWidget = None):
    box = QGroupBox(parent)
    box.setFlat(True)
    box.setLayout(QVBoxLayout())
    box.layout().addWidget(QLabel("<h2>brainglobe-registration</h2>"))
    subbox = QGroupBox(parent)
    subbox.setFlat(True)
    subbox.setLayout(QHBoxLayout())
    subbox.layout().setSpacing(0)
    subbox.layout().setContentsMargins(0, 0, 0, 0)
    subbox.setStyleSheet("border: 0;")
    subbox.layout().addWidget(_logo_widget(parent=box))
    subbox.layout().addWidget(_docs_links_widget(parent=box))
    box.layout().addWidget(subbox)
    return box
