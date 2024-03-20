from agent_data.agent_data.ontology import AgentAction
from . import metric_registry

@metric_registry.registry('Action')
class Action:
    def __init__(self, text: str, goal_text: str):
        self.action = AgentAction.from_text(text)
        self.goal_action = AgentAction.from_text(goal_text)

    def is_right_action(self) -> bool:
        return self.action.action_prefix == self.goal_action.action_prefix

    def is_right_first_command(self) -> bool:
        if not self.goal_action.commands:
            return not self.action.commands
        if not self.action.commands:
            return False
        return self.action.commands[0] == self.goal_action.commands[0]

    def is_right_status(self) -> bool:
        return self.action.status == self.goal_action.status