from .caching import *  # noqa
from .logging import *  # noqa

DEFAULT_TIME_FORMAT = "YYYY-MM-DD HH:mm:ss.SSS ZZ"


def leftpad(thing, n: int = 2, char=" "):
    return "\n".join([n * char + line for line in str(thing).splitlines()])
