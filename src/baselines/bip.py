"""BIP adapter. Requires a separately installed upstream bip package."""
def pi_analytical(*args,**kwargs):
    try:
        from bip.core_utils import pi_analytical as impl
    except ImportError as exc:
        raise ImportError("BIP is optional; see docs/baselines.md for installation.") from exc
    return impl(*args,**kwargs)

def get_all_zs(*args,**kwargs):
    try:
        from bip.utils import get_all_zs as impl
    except ImportError as exc:
        raise ImportError("BIP is optional; see docs/baselines.md for installation.") from exc
    return impl(*args,**kwargs)
