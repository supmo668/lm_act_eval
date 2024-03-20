from .actions import Action
from typing import List

from . import metric_registry

@metric_registry.registry('Trajectory')
class Trajectory(Action, object):
    def __init__(
      self, action_texts: List[str], goal_texts: List[str]):
      """
      Initialize the object with action and goal texts.

      :param action_texts: List of strings representing actions.
      :param goal_texts: List of strings representing goal actions.
      """
      self.actions = []
      self.goal_actions = []
      for text, goal_text in zip(action_texts, goal_texts):
        self.actions.append(Action.from_text(text))
        self.goal_actions.append(Action.from_text(goal_text)) 

    def is_right_actions(self) -> bool:
        traj_right_action = []
        for action, goal_action in zip(self.actions, self.goal_actions):
            self.__setattr__('action', action)
            self.__setattr__('goal_action', goal_action)
            traj_right_action.append(self.is_right_action)
        return traj_right_action

    def is_right_commands(self) -> bool:
        traj_right_command = []
        for action, goal_action in zip(self.actions, self.goal_actions):
            self.__setattr__('action', action)
            self.__setattr__('goal_action', goal_action)
            traj_right_command.append(self.is_right_first_command)
        return traj_right_command

    def is_right_statuses(self) -> bool:
        traj_right_status = []
        for action, goal_action in zip(self.actions, self.goal_actions):
            self.__setattr__('action', action)
            self.__setattr__('goal_action', goal_action)
            traj_right_status.append(self.is_right_status)
        return traj_right_status