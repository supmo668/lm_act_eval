
from pyclbr import Class
from typing import Callable, Union, Type

class Registry(dict):
    def register(self, name):
        """
        Registers a function with the given name in the dictionary.

        Parameters:
            name (str): The name of the function to be registered.

        Returns:
            function: The registered function/class

        """
        def inner(func: Class | Callable):
            self[name] = func
            return func
        return inner

    def get(self, name):
        try:
            return self[name]
        except KeyError:
            raise Exception(f"{name} don't exist in registry.")

    @classmethod
    def list_registered(cls):
        """
        Lists all registered names in the registry.

        Returns:
            list: A list of names of all registered functions or classes.
        """
        return list(cls.keys())