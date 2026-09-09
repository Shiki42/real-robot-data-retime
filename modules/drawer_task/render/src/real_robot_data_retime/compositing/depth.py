"""Read original aligned metric depth with the RGB registration transform."""

from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np


class AlignedDepth:
    def __init__(self, root, references, transforms, shape):
        self.root = Path(root).resolve()
        self.references = references
        self.transforms = np.asarray(transforms)
        self.shape = tuple(shape)
        if len(references) != len(transforms):
            raise ValueError("depth references must match RGB frames")
        self.cache = OrderedDict()
        self.archives = {}

    def __call__(self, index):
        index = int(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        ref = self.references[index]
        path = ref["path"]
        if not path.startswith("zip://") or "#" not in path:
            raise ValueError("expected zip-backed source depth")
        archive, member = path[6:].split("#", 1)
        archive = (self.root / archive).resolve()
        if not archive.is_relative_to(self.root):
            raise ValueError("depth archive escapes dataset root")
        if archive not in self.archives:
            self.archives[archive] = ZipFile(archive)
        encoded = self.archives[archive].read(member)
        depth = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError("expected uint16 millimetre depth image")
        h, w = self.shape
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.warpAffine(
            depth,
            self.transforms[index, :2],
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.cache[index] = depth
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return depth

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()
        self.cache.clear()
