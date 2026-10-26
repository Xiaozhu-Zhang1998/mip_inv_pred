"""External causalicp adapter."""
def fit(*args,**kwargs):
    try:
        import causalicp
    except ImportError as exc:
        raise ImportError("Install causalicp for ICP experiments; see docs/baselines.md.") from exc
    return causalicp.fit(*args,**kwargs)
