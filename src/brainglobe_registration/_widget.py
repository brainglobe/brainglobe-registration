from typing import TYPE_CHECKING

from magicgui import magic_factory

from bg_elastix.elastix.register import run_registration

if TYPE_CHECKING:
    import napari


@magic_factory
def register(
    viewer: "napari.Viewer",
    image: "napari.layers.Image",
    atlas_image: "napari.layers.Image",
    rigid=True,
    affine=True,
    bspline=True,
    affine_iterations="2048",
    log=False,
):
    result, parameters = run_registration(
        atlas_image.data,
        image.data,
        rigid=rigid,
        affine=affine,
        bspline=bspline,
        affine_iterations=affine_iterations,
        log=log,
    )
    viewer.add_image(result, name="Registered Image")
