"""Loads control/ the way LiteLLM does inside the cluster, but without LiteLLM.

In the cluster `control/*.py` is mounted as the package `security` (and inspect.py as `security_inspect`), and
the code imports `security.fleet_state` / `security.places`. Here the same names point at the files in
control/, and the two LiteLLM classes the hooks inherit from are replaced by plain stubs.
control/ is NOT put on sys.path: its inspect.py would shadow the standard library module of the same name.
"""
import importlib
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL = os.path.join(ROOT, "control")


def _stub_litellm():
    if "litellm.integrations.custom_logger" in sys.modules:
        return
    for name in ("litellm", "litellm.integrations", "litellm.integrations.custom_logger",
                 "litellm.proxy", "litellm.proxy._types"):
        sys.modules[name] = types.ModuleType(name)

    class CustomLogger:           # base class of the hooks
        pass

    class UserAPIKeyAuth:         # only used as a type annotation
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    sys.modules["litellm.integrations.custom_logger"].CustomLogger = CustomLogger
    sys.modules["litellm.proxy._types"].UserAPIKeyAuth = UserAPIKeyAuth


def _alias_security_package():
    package = types.ModuleType("security")
    package.__path__ = [CONTROL]
    sys.modules["security"] = package


def _load_guard():
    spec = importlib.util.spec_from_file_location("security_inspect", os.path.join(CONTROL, "inspect.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["security_inspect"] = module
    spec.loader.exec_module(module)
    return module


_stub_litellm()
_alias_security_package()
fleet_state = importlib.import_module("security.fleet_state")
place = importlib.import_module("security.place")
admission = importlib.import_module("security.admission")
guard = _load_guard()
