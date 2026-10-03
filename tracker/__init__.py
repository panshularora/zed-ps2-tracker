"""Framework-free (NumPy/SciPy) online multi-object tracker core.

Everything in this package is causal: ``update()`` only ever sees the current frame's detections
and the tracker's own state. No future frames, interpolation or global linking.
"""

__version__ = "0.1.0"
