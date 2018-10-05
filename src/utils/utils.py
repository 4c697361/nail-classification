import os

import settings

global rootdir
rootdir = settings.PROJECT_ROOT


class dirs:
    base_dir = os.path.join(rootdir, "data")
    original_dataset_dir = os.path.join(base_dir, "raw/nailgun")
    train_dir = os.path.join(base_dir, "processed/train")
    validation_dir = os.path.join(base_dir, "processed/validate")
    test_dir = os.path.join(base_dir, "processed/test")
    model_dir = os.path.join(rootdir, "models")


class params:
    batch_size = 6
    epochs = 12
    learning_rate = 0.001
    image_width = 150
    image_heigth = 150
    croped_size = (590, 150, 830, 900)   # (xmin, ymin, dx, dy) for crop step 1
    cropwidth = 350     # image sidelenght for crop step 2


def main():
    pass


if __name__ == '__main__':
    main()
