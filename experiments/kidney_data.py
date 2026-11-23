import pickle
import pickletools
from pathlib import Path

import numpy as np


LABELS = ("Normal", "Mild", "Moderate", "Severe", "Excluded")
VALUES = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [0, 0]])
DATA_OPS = set("PROTO FRAME EMPTY_LIST MEMOIZE MARK EMPTY_DICT SHORT_BINUNICODE BINUNICODE BININT1 BININT2 BININT BINGET LONG_BINGET SETITEM SETITEMS APPEND APPENDS STOP".split())


def load_panel(root, *, labels_only=False):
    raters = []
    for reader in (1, 2, 3):
        payload = (root / f"expert{reader}.pickle").read_bytes()
        if any(op.name not in DATA_OPS for op, _, _ in pickletools.genops(payload)):
            raise ValueError(f"Unexpected pickle operation in reader {reader}")
        records = pickle.loads(payload)
        by_image = {}
        for row in records:
            image = Path(row["image"])
            if image.parent != Path("glomeruli-patches", row["slide"]) or image in by_image:
                raise ValueError(f"Invalid or repeated image in reader {reader}: {image}")
            if not image.name.startswith(f"{row['slide']}-{row['index']}-"):
                raise ValueError(f"Image metadata mismatch: {image}")
            by_image[image] = LABELS.index(row["label"])
        raters.append(by_image)
    if labels_only:
        frame = raters[0].keys()
        if not frame or any(rater.keys() != frame for rater in raters[1:]):
            raise ValueError("Label-only analysis requires three identical, nonempty crop lists")
        images = sorted(frame)
    else:
        images = sorted(path.relative_to(root) for path in root.glob("glomeruli-patches/*/*.png"))
        if any(set(rater) - set(images) for rater in raters):
            raise ValueError("At least one annotation refers to an absent image")
    slides = np.array([image.parent.name for image in images])
    grades = np.array([[rater.get(image, -1) for rater in raters] for image in images])
    return images, slides, grades
