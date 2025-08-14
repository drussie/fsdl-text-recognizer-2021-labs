# lab3/text_recognizer/data/__init__.py
"""
Lightweight re-exports for data classes.

We avoid importing submodules that may have extra, optional deps so that
`import text_recognizer.data` succeeds. Each export is done behind a try/except.
"""

__all__ = []

def _safe_export(module_name: str, *names: str) -> None:
    try:
        mod = __import__(f"{__name__}.{module_name}", fromlist=list(names))
        g = globals()
        for n in names:
            g[n] = getattr(mod, n)
            __all__.append(n)
    except Exception:
        # Silently skip missing/optional pieces
        pass

# Only export the pieces the training script actually expects to find here
_safe_export("base_data_module", "BaseDataModule")
_safe_export("mnist", "MNIST")
_safe_export("emnist", "EMNIST", "EMNISTLines")
_safe_export("sentence_generator", "SentenceGenerator")

# If you really have base_dataset.py and want to re-export it, uncomment:
# _safe_export("base_dataset", "BaseDataset")
