def __getattr__(name):
    if name == "SU2_Runner":
        from .run import SU2_Runner

        return SU2_Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SU2_Runner", # Handles the execution of SU2, and the management of its input/output files
]
