# Extracted from identifiability-research-1, src/idiff/types.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3 (sha256 dbeb051b8928a90d56eeeb6feffa6d37cbaee289f14780926db05319942d3482).
# Modified in this repository: unchanged apart from this notice.
# No upstream TCL ancestor; this is a project addition, not a TCL-derived file.
# See docs/provenance.md.
"""Array aliases shared by the numerical API (float64 outside torch)."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
