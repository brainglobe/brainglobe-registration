from typing import TYPE_CHECKING, List

from magicgui import magic_factory, magicgui
from magicgui.widgets import FunctionGui, PushButton
from napari.layers.image import Image
from napari.utils.events import Event
import napari

from bg_elastix.elastix.register import run_registration, get_atlas_by_name
from bg_atlasapi.list_atlases import get_downloaded_atlases

if TYPE_CHECKING:
    import napari


def register_widget() -> FunctionGui:
    @magicgui(
        # widget_init=_on_init_register,
        atlas_name={"choices": get_downloaded_atlases()},
        translate_x=dict(widget_type="SpinBox", min=-5000, max=5000),
        translate_y=dict(widget_type="SpinBox", min=-5000, max=5000),
        get_atlas_button=dict(widget_type="PushButton", text="Get Atlas"),
        move_sample_button=dict(widget_type="PushButton", text="Translate Sample")
    )
    def register(
            viewer: "napari.Viewer" = None,
            atlas_name=None,
            get_atlas_button=False,
            image_to_adjust: "napari.layers.Image" = None,
            translate_x: int = 0,
            translate_y: int = 0,
            move_sample_button=False,
            image: "napari.layers.Image" = None,
            atlas_image: "napari.layers.Image" = None,
            rigid=True,
            affine=True,
            bspline=True,
            use_default_params=True,
            affine_iterations="2048",
            log=False,
    ) -> List[napari.layers.Layer]:
        current_atlas_slice = viewer.dims.current_step[0]

        result, parameters = run_registration(
            atlas_image.data[current_atlas_slice, :, :],
            image.data,
            rigid=rigid,
            affine=affine,
            bspline=bspline,
            use_default_params=use_default_params,
            affine_iterations=affine_iterations,
            log=log,
        )

        return [napari.layers.Image(result, name="registered image"),
                napari.layers.Labels(viewer.layers[-1].data[current_atlas_slice, :, :], name="registered annotations")]

    register.rigid.value = False

    @register.move_sample_button.changed.connect
    def move_atlas_button_click():
        delta_x = getattr(register, "translate_x").value
        delta_y = getattr(register, "translate_y").value
        image: "napari.layers.Image" = getattr(register, "image_to_adjust").value

        curr_location = image.translate

        image.translate = (curr_location[0] + delta_y, curr_location[1] + delta_x)

    @register.get_atlas_button.changed.connect
    def get_atlas_button_click():
        atlas_name = getattr(register, "atlas_name").value
        viewer = getattr(register, "viewer").value
        atlas = get_atlas_by_name(atlas_name)

        viewer.add_image(atlas.reference, name=atlas_name, colormap="red", opacity=0.6, blending="translucent")
        viewer.add_labels(atlas.annotation, name=atlas_name + "Annotation", visible=False)
        viewer.grid.enabled = False

    return register
