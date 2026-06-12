"""
Hydra SearchPathPlugin for the wheat3dgs repo.

Each app (mask_generation, reconstruction) points its Hydra config_path INTO its own
config folder (e.g. configs/mask_generation) so its config groups get clean names —
`method=sahi_yolo_sam` instead of the namespaced `mask_generation/method@method=...`.
That alone would hide the SHARED `configs/dataset/` group, so this plugin adds the
top-level `configs/` folder to Hydra's search path, making the shared groups
(dataset, …) reachable from every app.

Portable: the configs path is derived from THIS file's location (__file__), so it
works wherever the repo is checked out — local WSL or the Euler cluster — with no
hardcoded absolute path. Hydra auto-discovers this because `hydra_plugins` is a
namespace package on sys.path (src/ is on the path via the editable install).
"""

import os
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class WheatSearchPathPlugin(SearchPathPlugin):
    """Add the repo-root configs/ folder to Hydra's search path so apps that root their
    config_path inside their own subfolder can still find the shared groups (dataset, ...)."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # this file lives at <repo>/src/hydra_plugins/wheat_searchpath/__init__.py → 3 up = repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        configs_dir = os.path.join(repo_root, "configs")
        search_path.append(provider="wheat-shared-configs", path="file://" + configs_dir)
