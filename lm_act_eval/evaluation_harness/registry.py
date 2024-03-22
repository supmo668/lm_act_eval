
from pyclbr import Class
from typing import Callable, Union, Type, List

import warnings

class Registry(dict):
    # Keeping a class-level registry to track all registered items
    _registry = {}
    
    def register(self, name: str) -> Callable:
        def inner(func: Union[Type, Callable]) -> Union[Type, Callable]:
            self.__class__._registry[name] = func
            self[name] = func  # Keep instance-specific registration as well
            return func
        return inner

    def get(self, name: Union[str, List[str]]):
        if isinstance(name, list):
            # Prepare a result dictionary for names that exist
            result = {}
            for n in name:
                try:
                    result[n] = self[n]
                except KeyError:
                    warnings.warn(f"{n} doesn't exist in registry.")
            return result
        else:
            # Single name case, as before
            try:
                return self[name]
            except KeyError:
                raise Exception(f"{name} doesn't exist in registry.")

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())