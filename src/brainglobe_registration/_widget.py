from typing import TYPE_CHECKING, List

from magicgui import magicgui
from magicgui.widgets import FunctionGui
from bg_atlasapi.list_atlases import get_downloaded_atlases

from brainglobe_registration.elastix.register import (
    run_registration,
    get_atlas_by_name,
)

if TYPE_CHECKING:
    import napari


def register_widget() -> FunctionGui:
    @magicgui(
        # widget_init=_on_init_register,
        atlas_name={"choices": get_downloaded_atlases()},
        translate_x=dict(widget_type="SpinBox", min=-5000, max=5000),
        translate_y=dict(widget_type="SpinBox", min=-5000, max=5000),
        rotate=dict(widget_type="SpinBox", min=-180, max=180),
        get_atlas_button=dict(widget_type="PushButton", text="Get Atlas"),
        start_alignment_button=dict(
            widget_type="PushButton", text="Start Alignment"
        ),
        move_sample_button=dict(widget_type="PushButton", text="Adjust Image"),
        reset_button=dict(widget_type="PushButton", text="Reset Image"),
    )
    def register(
        viewer: "napari.Viewer" = None,
        atlas_name=None,
        get_atlas_button=False,
        image_to_adjust: "napari.layers.Image" = None,
        start_alignment_button=False,
        translate_x: int = 0,
        translate_y: int = 0,
        rotate: int = 0,
        move_sample_button=False,
        reset_button=False,
        image: "napari.layers.Image" = None,
        atlas_image: "napari.layers.Image" = None,
        rigid=True,
        affine=True,
        bspline=False,
        use_default_params=True,
        affine_iterations="2048",
        log=False,
    ) -> List["napari.layers.Layer"]:
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

        return [
            napari.layers.Image(result, name="registered image"),
            napari.layers.Labels(
                viewer.layers[-1].data[current_atlas_slice, :, :],
                name="registered annotations",
            ),
        ]

    @register.get_atlas_button.changed.connect
    def get_atlas_button_click():
        atlas = get_atlas_by_name(register.atlas_name.value)

        register.viewer.value.add_image(
            atlas.reference,
            name=register.atlas_name.value,
            colormap="red",
            opacity=0.6,
            blending="translucent",
        )
        register.viewer.value.add_labels(
            atlas.annotation,
            name=register.atlas_name.value + "Annotation",
            visible=False,
        )
        register.viewer.value.grid.enabled = True

    @register.move_sample_button.changed.connect
    def move_atlas_button_click():
        if register.image_to_adjust.value:
            register.image_to_adjust.value.translate = (
                register.translate_y.value,
                register.translate_x.value,
            )
            register.image_to_adjust.value.rotate = register.rotate.value

    @register.reset_button.changed.connect
    def reset_button_on_click():
        if register.image_to_adjust.value:
            register.image_to_adjust.value.translate = (0, 0)
            register.image_to_adjust.value.rotate = 0

            register.translate_x.value = 0
            register.translate_y.value = 0
            register.rotate.value = 0

    @register.start_alignment_button.changed.connect
    def start_alignment_button_on_clik():
        if register.image_to_adjust.value:
            image_to_adjust: "napari.layers.Image" = (
                register.image_to_adjust.value
            )
            image_to_adjust.opacity = 0.6
            image_to_adjust.colormap = "green"
            image_to_adjust.blending = "translucent"

    return register
