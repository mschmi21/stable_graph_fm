from src.paths.linear_path import LinearPath
from src.paths.topological_path import TopologicalPath


def get_path(config, L, coeffs = None):

    name = config["path"]["name"]

    if name == "linear":
        return LinearPath()

    if name == "topological":
        return TopologicalPath(L = L, coeffs = coeffs)

    raise ValueError(f"Unknown path: {name}")