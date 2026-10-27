class SpecError(ValueError):
    """A JobSpec is malformed, unsafe, or outside the supported contract."""


class BundleError(ValueError):
    """A compiled bundle is corrupt or violates the deterministic bundle contract."""


class NotebookError(BundleError):
    """Notebook bytes or embedded payload cells violate the notebook contract."""


class DriveTransportError(BundleError):
    """Drive staging, controller, receipt, or cleanup state is invalid."""
