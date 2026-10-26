"""WVM adapter. Requires separately downloaded upstream WVM code."""
def load_wvm():
    try:
        from WVM.py_implementation.wasserstein_variance import DirectWassersteinVariance,WV_screening
    except ImportError as exc:
        raise ImportError("WVM is optional; see docs/baselines.md for installation.") from exc
    return DirectWassersteinVariance,WV_screening

def DirectWassersteinVariance(*args,**kwargs):
    impl,_=load_wvm()
    return impl(*args,**kwargs)

def WV_screening(*args,**kwargs):
    _,impl=load_wvm()
    return impl(*args,**kwargs)
