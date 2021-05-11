

import os

from ..core import load_operation_module

# Load core module before loading extensions
load_operation_module(os.path.join(os.path.dirname(__file__), "libpp_geometry.so"))

# Import Python extensions
from . import pp_geometry_py
