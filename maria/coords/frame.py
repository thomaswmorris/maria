import os

from maria.utils.io import read_yaml

here, this_filename = os.path.split(__file__)

FRAMES = read_yaml(f"{here}/frame.yml")


def parse_frame(frame):
    for key, config in FRAMES.items():
        if frame in [key, *config["aliases"]]:
            return key
    raise ValueError(f"Invalid frame '{frame}'")


class Frame:
    def __init__(self, frame):
        if isinstance(frame, type(self)):
            frame = frame.name
        self.name = parse_frame(frame)

    def __getattr__(self, key, delim="_"):
        res = FRAMES[self.name]
        while key:
            if key in res:
                return res[key]
            else:
                parts = key.split(delim)
                res = res.get(parts[0], {})
                key = delim.join(parts[1:])
        raise AttributeError(f"'{type(self)}' object has no attribute '{key}'")

    def __repr__(self):
        return f"{type(self).__name__}('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other

    def __copy__(self):
        return Frame(self.name)

    def __deepcopy__(self, memo):
        return self.__copy__()
