from pathlib import Path
import mlx.data as dx
# import rasterio
# from rasterio.plot import show


# image = rasterio.open('/Users/danielpetterson/GitHub/xView3-SAR-ship-detection/trainingset/72dba3e82f782f67t/VV_dB.tif')
# show(image)
# print(image.width)
# print(image.height)

# A simple python function returning a list of dicts. All samples in MLX data
# are dicts of arrays.
def files_and_classes(root: Path):
    files = [str(f) for f in root.glob("**/*.jpg")]
    files = [f for f in files if "BACKGROUND" not in f]
    classes = dict(
        map(reversed, enumerate(sorted(set(f.split("/")[-2] for f in files))))
    )

    return [
        dict(image=f.encode("ascii"), label=classes[f.split("/")[-2]]) for f in files
    ]

# alternative option
def files_and_classes(root: Path):
    """Load the files and classes from an image dataset that contains one folder per class."""
    images = list(root.rglob("*.jpg"))
    categories = [p.relative_to(root).parent.name for p in images]
    category_set = set(categories)
    category_map = {c: i for i, c in enumerate(sorted(category_set))}

    return [
        {
            "image": str(p.relative_to(root)).encode("ascii"),
            "category": c,
            "label": category_map[c]
        }
        for c, p in zip(categories, images)
    ]


dset = (
    # Make a buffer (finite length container of samples) from the python list
    dx.buffer_from_vector(files_and_classes(root))

    # Shuffle and transform to a stream
    .shuffle()
    .to_stream()

    # Implement a simple image pipeline. No random augmentations here but they
    # could be applied.
    # Can use Buffer.pad which may replace pad function in reference.
    .load_image("image")  # load the file pointed to by the 'image' key as an image
    .image_resize_smallest_side("image", 256)
    .image_center_crop("image", 224, 224)
    #.key_transform("image", normalize)

    # Accumulate into batches
    .batch(batch_size)

    # Cast to float32 and scale to [0, 1]. We do this in python and we could
    # have done any transformation we could think of.
    .key_transform("image", lambda x: x.astype("float32") / 255)

    # Finally, fetch batches in background threads
    .prefetch(prefetch_size=8, num_threads=8)
)

# dset is a python iterable so one could simply
for sample in dset:
    # access sample["image"] and sample["label"]
    pass