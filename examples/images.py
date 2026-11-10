"""Process the images bundled with the repository.

The example does not require the Python OpenCV package; image decoding and
augmentation are performed by the PixelPipes native module.
"""

import argparse
from pathlib import Path

from pixelpipes import Engine
from pixelpipes.image.augmentation import ImageBrightness
from pixelpipes.image.geometry import RandomPatchView, ViewImage
from pixelpipes.numbers import SampleUnform
from pixelpipes.resource.list import RandomResource
from pixelpipes.resource.loading import ImageDirectory
from pixelpipes.utilities import pipeline


@pipeline()
def image_stream(directory):
    images = ImageDirectory(path=directory, channels="RGB", depth="Char")
    source = RandomResource(images)["image"]
    view = RandomPatchView(source, width=64, height=64)
    patch = ViewImage(source, view, width=64, height=64)
    return ImageBrightness(patch, SampleUnform(-20, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("images"),
        help="directory containing input images",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    batch = Engine(
        image_stream(str(args.directory)),
        batch_size=args.batch,
        workers=args.workers,
    ).next_batch()[0]
    print("image batch shape:", batch.shape)
    print("value range:", int(batch.min()), int(batch.max()))


if __name__ == "__main__":
    main()
