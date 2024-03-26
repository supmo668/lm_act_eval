

from autoevals.llm import *
from autoevals.string import Levenshtein
import warnings

from typing import Dict, List, Tuple

from beartype import beartype

from lm_act_eval.evaluation_harness.evaluators.webarena_rl.base import Evaluator, Trajectory

from typing import List

from lm_act_eval.evaluation_harness.evaluators.registry import metric_registry

from lm_act_eval.ontology.agent import AgentAction

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
      
@metric_registry.register('trajectory')
class TrajectoryEvaluator(Evaluator):
    """Check if the numerical relationship is correct"""
    @staticmethod
    @beartype
    def edit_distance(ref: str, pred: str) -> float:
      """
      Calculate the edit distance between two strings.

      Parameters:
        ref (str): The reference string.
        pred (str): The predicted string.

      Returns:
        float: The edit distance between the reference and predicted strings.
      """
      return TrajectoryEvaluator.levenshtein_comparator(ref, pred).score

    def __call__(
        self,
        trajectory: Tuple[List[List], List[List]],
        metrics: list,
        # page: Page | PseudoPage | None = None,
        # client: CDPSession | None = None,
      ) -> Dict[str, int]:
        """
        Call method to calculate the score based on the given trajectory and metrics. 

        Args:
            action trajectory (List[Tuple[List, List]]): The action trajectory data.
            metrics (list): The list of metrics to calculate score.
            page (Page | PseudoPage | None, optional): The page type. Defaults to None.
            client (CDPSession | None, optional): The client session. Defaults to None.

        Returns:
            Dict[str, int]: The calculated score for each approach.
        """
        score = dict()
        for approach in metrics:
            match approach:
                case "exact_match":
                    score.update(
                      {approach: self.edit_distance(**trajectory)}
                    )