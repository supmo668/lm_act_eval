from lm_act_eval.evaluation_harness.helper_functions.multion import (
  action_prefix,
  clean_extracted_text,
  extract_thought,
  extract_action,
  extract_explanation,
  ParseChatCompletion
)

# fail-safe functions
process_fs = {
  "action": lambda x: extract_action(x) if type(x) == str else "",
  "thought": lambda x: extract_thought(x) if type(x) == str else "",
  "explanation": lambda x: extract_explanation(x) if type(x) == str else ""
}
