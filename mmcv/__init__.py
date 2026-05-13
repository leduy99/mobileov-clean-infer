"""Minimal mmcv shim for Registry used by bundled SANA code."""

try:
    from mmengine.registry import Registry  # type: ignore
except Exception:
    class Registry:
        """Minimal Registry supporting register_module/build."""

        def __init__(self, name: str):
            self._name = name
            self._module_dict = {}

        def register_module(self, module=None, name=None):
            def _register(cls):
                key = name or cls.__name__
                self._module_dict[key] = cls
                return cls

            return _register(module) if module is not None else _register

        def build(self, cfg, default_args=None):
            if not isinstance(cfg, dict):
                raise TypeError("cfg must be a dict")
            cfg = cfg.copy()
            obj_type = cfg.pop("type")
            if isinstance(obj_type, str):
                if obj_type not in self._module_dict:
                    raise KeyError(f"{obj_type} is not registered in {self._name}")
                obj_cls = self._module_dict[obj_type]
            else:
                obj_cls = obj_type
            if default_args:
                for k, v in default_args.items():
                    cfg.setdefault(k, v)
            return obj_cls(**cfg)

def build_from_cfg(cfg, registry, default_args=None):
    """Compatibility wrapper used by bundled SANA code."""
    return registry.build(cfg, default_args=default_args)

__all__ = ["Registry", "build_from_cfg"]
