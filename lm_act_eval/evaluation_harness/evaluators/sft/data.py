from lm_act_eval.evaluation_harness.helper_functions.utils import function_registry

from typing import *
def cfg_to_function(funct_pairs: Dict) -> Generator[Tuple[Tuple[str, str], str], None, None]:
  """
  A function that processes a function query along with a function pair and returns a tuple containing a source field, target field, and function name.
  
  Parameters:
    function_query (Dict): A dictionary containing the function query.
    funct_pair (List[Dict]): A list of dictionaries containing function pairs.
  
  Returns:
    Tuple[Tuple[str, str], str]: A tuple containing a tuple with source field and target field, and a string representing the function name.
  """
  for tgt_field, function_query in funct_pairs.items():
    # field_name: str
    # function_query: Dict
    src_field, func_name = dict(function_query).popitem()
    if not func_name: func_name = "DO NOTHING"
    if '.' in func_name: 
      _function_is_cls = True
      
    else: _function_is_cls = False
    # class involve
    if _function_is_cls:
      func_name, cls_func = func_name.split(".")
    # get function/class
    function = function_registry.get(func_name)
    if _function_is_cls:
      # retrieve from class
      function = getattr(function(), cls_func)
    yield (src_field, tgt_field), function