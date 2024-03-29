
from pyclbr import Class
from typing import Callable, Union, Type, List

import warnings

class Registry:
    def __init__(self):
        """
        Initializes an instance of the class with an instance-specific registry to track all registered items.
        """
        # Initialize an instance-specific registry to track all registered items
        self._registry = {}
    
    def register(self, name: str) -> Callable:
        def inner(func: Union[Type, Callable]) -> Union[Type, Callable]:
            # Update the instance-specific registry
            self._registry[name] = func
            return func
        return inner

    def get(self, name: Union[str, List[str]]):
        """
        Retrieves the value associated with the given name from the registry.

        Parameters:
            name (Union[str, List[str]]): The name or names of the values to retrieve.

        Returns:
            Union[Dict[str, Any], Any]: If `name` is a list, a dictionary containing the values associated with each name that exists in the registry. If `name` is a single string, the value associated with the name in the registry.

        Raises:
            Exception: If `name` is a single string and it doesn't exist in the registry.
            UserWarning: If `name` is a list and any of the names in the list don't exist in the registry.
        """
        if isinstance(name, list):
            # Prepare a result dictionary for names that exist
            result = {}
            for n in name:
                if n in self._registry:
                    result[n] = self._registry[n]
                else:
                    warnings.warn(f"{n} doesn't exist in the registry.")
            return result
        else:
            # Single name case
            if name in self._registry:
                return self._registry[name]
            else:
                raise Exception(f"{name} doesn't exist in the registry.")

    def list_registered(self):
        # List registered items for this instance
        return list(self._registry.keys())
    
metric_registry = Registry()

evaluator_registry = Registry()