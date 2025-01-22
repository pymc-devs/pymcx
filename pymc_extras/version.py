import os


def get_version():
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version.txt")
    with open(version_file) as f:
        version = f.read().strip()
    return version


__version__ = get_version()
