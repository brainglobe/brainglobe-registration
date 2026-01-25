Before submitting a pull request (PR), please read the [contributing guide](https://github.com/brainglobe/.github/blob/main/CONTRIBUTING.md).

Please fill out as much of this template as you can, but if you have any problems or questions, just leave a comment and we will help out :)

## Description

**What is this PR**

- [ ] Bug fix
- [x] Addition of a new feature
- [ ] Other

**Why is this PR needed?**

Users need a quick way to undo scaling changes and restore the moving image to its original state.

**What does this PR do?**

Adds a "Reset Moving Image" button under "Scale Moving Image" that emits a reset signal and restores the moving image data from `_moving_image_data_backup` in the registration widget.

## References

- Issue #116

## How has this PR been tested?

Manual test in napari: scale a moving image, then click "Reset Moving Image" to restore the original shape/data.

## Is this a breaking change?

No.

## Does this PR require an update to the documentation?

No.

## Checklist:

- [ ] The code has been tested locally
- [ ] Tests have been added to cover all new functionality (unit & integration)
- [ ] The documentation has been updated to reflect any changes
- [ ] The code has been formatted with [pre-commit](https://pre-commit.com/)
