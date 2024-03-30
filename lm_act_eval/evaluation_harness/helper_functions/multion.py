import json
import re
import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Dict

from beartype import beartype
import pandas as pd
from datasets import Dataset

from .utils import function_registry

COMMANDS_PREFIX = "COMMANDS:"
ANSWER_PREFIX = "ANSWER:"
ASK_USER_HELP_PREFIX = "ASK_USER_HELP:"
EXPLANATION_PREFIX = "EXPLANATION:"
STATUS_PREFIX = "STATUS:"


def action_prefix(action: str) -> Optional[str]:
    for prefix in [COMMANDS_PREFIX, ANSWER_PREFIX, ASK_USER_HELP_PREFIX]:
        if prefix in action:
            return prefix
    return None

@function_registry.register('clean')
def clean_extracted_text(text: str) -> str:
    return text.strip().strip(r"\n")

@function_registry.register('extract_first')
def extract_first(text:str, term:str="") -> str:
    return re.findall(rf'{term}:\s*(\d+)', text)[0] if re.search(rf'{term}:\s*(\d+)', text) else None

@function_registry.register('extract_thought')
@beartype
def extract_thought(text: str) -> str:
    print(text)
    match = re.search(
        r"(.*?)(COMMANDS:|ANSWER:|ASK_USER_HELP:|EXPLANATION:|STATUS:|$)",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    return clean_extracted_text(match.group(1))

@function_registry.register('extract_action')
def extract_action(text: str) -> str:
    """
    Extracts the action from the given text.

    Args:
        text (str): The text from which the action needs to be extracted.

    Returns:
        str: The extracted action.

    """
    match = re.search(
        r"(COMMANDS:|ANSWER:|ASK_USER_HELP:)(.*?)(EXPLANATION:|STATUS:|$)",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    # Modified to return only the command without prefix
        # e.g. clean_extracted_text(match.group(1) + match.group(2))
    return clean_extracted_text(match.group(2))

@function_registry.register('extract_explanation')
def extract_explanation(text: str) -> str:
    match = re.search(
        r"(EXPLANATION:)(.*?)(STATUS:|COMMANDS:|ANSWER:|ASK_USER_HELP:|$)",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    return clean_extracted_text(match.group(1) + match.group(2))

@function_registry.register('extract_status')
def extract_status(text: str) -> str:
    match = re.search(
        r"(STATUS:)(.*?)(COMMANDS:|ANSWER:|ASK_USER_HELP:|EXPLANATION:|$)",
        text,
        re.DOTALL,
    )
    if not match:
        return ""
    return clean_extracted_text(match.group(1) + match.group(2))

@function_registry.register('extract_commands')
def extract_commands(action: str) -> list[str]:
    if COMMANDS_PREFIX in action:
        objective_start = action.find(COMMANDS_PREFIX) + len(COMMANDS_PREFIX)
        objective_end = len(action)
        return action[objective_start:objective_end].strip().split("\n")
    return []

@function_registry.register('parse_completion')
class ParseChatCompletion:
    """
    Fix & parse malformed json from chat_completion & provide fallback
    """
    def _extract_content_value(self, input_str: str):
        pattern = r"'content': '(.*?)'}"
        matches = re.findall(pattern, input_str, re.DOTALL)
        if not matches:
            return ""
        return matches[0]
    
    def parse_json(self, s: str, target_field=None) -> str | Dict:
        """
        Parse the input string to extract a JSON object. If successful, return the 'content' value from the JSON, or return "Content not provided" if not found. If the parsing fails, attempt to extract the content value using the extract_content_value method. If that also fails, return "NA".
        common target_field:
        """
        try:
            d = ast.literal_eval(s)
            json_str: str = json.dumps(d, indent=4)
            if target_field:
                return json.loads(json_str).get(target_field, '<Query not found>')
            else:
                return json.loads(json_str)
        except:
            try:
                return self._extract_content_value(s)
            except:
                return "NA"
            
    def parse_query(self, s:str) -> str:
        return self.parse_json('QUERY')
            
    def parse_content(self, s: str) -> str | Dict:
        return self.parse_json('chat_completion_messages',{[{}]})[0].get('content',"")
  

@dataclass
class AgentAction:
    thought: str
    action: str
    explanation: str
    status: str

    @property
    def commands(self) -> Optional[list[str]]:
        if self.action is None:
            return None
        return extract_commands(self.action)

    @property
    def text(self) -> str:
        return "\n\n".join(
            [self.thought, self.action, self.explanation, self.status]
        ).strip()

    @property
    def action_prefix(self) -> Optional[str]:
        return action_prefix(self.action)

    @classmethod
    def from_text(cls, text: str) -> "AgentAction":
        if text is None:
            return AgentAction("", "", "", "")
        thought = extract_thought(text)
        action = extract_action(text)
        assert action is not None
        explanation = extract_explanation(text)
        status = extract_status(text)
        return AgentAction(thought, action, explanation, status)

    def to_dict(self) -> dict:
        return dict(text=self.text, **asdict(self))


@dataclass
class State:
    objective: str
    user_context: str
    time: str
    action_explanation_history: list[str]
    url: Optional[str] = None
    dom: Optional[str] = None
    plan: Optional[str] = None
    additional_context: Optional[str] = None
    rules: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class StateActionDataset:
    def __init__(
        self,
        states: list[State],
        actions: list[AgentAction],
        rewards: Optional[list[float]] = None,
        trajectory_ids: Optional[list[str]] = None,
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.trajectory_ids = trajectory_ids
        records = [
            dict(**state.to_dict(), **action.to_dict())
            for state, action in zip(states, actions)
        ]
        if rewards is not None:
            for i, reward in enumerate(rewards):
                records[i]["reward"] = reward
        if trajectory_ids is not None:
            for i, trajectory_id in enumerate(trajectory_ids):
                records[i]["trajectory_id"] = trajectory_id
        self._df = pd.DataFrame.from_records(records)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def as_dataset(self) -> Dataset:
        return Dataset.from_pandas(self.df)

    def save(self, path: Path | str) -> None:
        self.df.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Path | str) -> "StateActionDataset":
        df = pd.read_csv(path)
        states = [
            State(
                row["objective"],
                row["user_context"],
                row["time"],
                row["action_explanation_history"],
                row["url"],
                row["dom"],
                row["plan"],
                row["additional_context"],
                row["rules"],
            )
            for _, row in df.iterrows()
        ]
        actions = [
            AgentAction(
                row["gen_short_thought"],
                row["gen_action"],
                row["gen_explanation"],
                row["gen_status"],
            )
            for _, row in df.iterrows()
        ]
        return cls(states, actions)


@dataclass
class Trajectory:
    states: list[State]
    actions: list[AgentAction]
    rewards: Optional[list[float]] = None
    success: Optional[bool] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return dict(
            states=[state.to_dict() for state in self.states],
            actions=[action.to_dict() for action in self.actions],
            rewards=self.rewards,
            success=self.success,
            metadata=self.metadata,
        )


class TrajectoryDataset:
    def __init__(self, trajectories: list[Trajectory]):
        self.trajectories = trajectories

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        trajectory_ids = [i for i, _ in enumerate(self.trajectories)]
        states = []
        actions = []
        rewards = []
        trajectory_ids = []
        for trajectory, i in zip(self.trajectories, trajectory_ids):
            states.extend(trajectory.states)
            actions.extend(trajectory.actions)
            if trajectory.rewards is not None:
                rewards.extend(trajectory.rewards)
            else:
                rewards.extend([None] * len(trajectory.states))
            trajectory_ids.extend([i] * len(trajectory.states))
        dataset = StateActionDataset(states, actions, rewards, trajectory_ids)
        dataset.save(path / "timesteps.csv")
        with open(path / "trajectories.json", "w") as f:
            json.dump(
                dict(
                    trajectories=[
                        dict(id=i, **trajectory.to_dict())
                        for i, trajectory in zip(trajectory_ids, self.trajectories)
                    ]
                ),
                f,
                indent=4,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: Path | str) -> "TrajectoryDataset":
        path = Path(path)
        with open(path / "trajectories.json", "r") as f:
            data = json.load(f)
        trajectories = [
            Trajectory(
                states=[State(**state) for state in trajectory["states"]],
                actions=[AgentAction(**action) for action in trajectory["actions"]],
                rewards=trajectory["rewards"],
                success=trajectory["success"],
                metadata=trajectory["metadata"],
            )
            for trajectory in data["trajectories"]
        ]
        return cls(trajectories)
