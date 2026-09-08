"""Exceptions shared by the pipeline and the renderer."""


class Cancelled(Exception):
    """Raised when the user cancels a running batch."""
