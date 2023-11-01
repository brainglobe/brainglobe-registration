# import numpy as np
#
# from bg_elastix.elastix import register
#
# # make_napari_viewer is a pytest fixture that returns a napari viewer object
# # capsys is a pytest fixture that captures stdout and stderr output streams
#
#
# def test_example_magic_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))
#
#     # this time, our widget will be a MagicFactory or FunctionGui instance
#     my_widget = register()
#
#     # if we "call" this object, it'll execute our function
#     my_widget(viewer.layers[0])
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == f"you have selected {layer}\n"
