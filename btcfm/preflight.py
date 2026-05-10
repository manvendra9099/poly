"""
Deprecated shim — canonical location is btcfm.runtime.preflight.

Kept so that any external code that imported from btcfm.preflight
continues to work without changes.
"""
from btcfm.runtime.preflight import (  # noqa: F401
    check_not_login_node,
    gpu_preflight,
    resolve_precision,
)
