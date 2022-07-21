from typing import Any, Dict, Iterator, List, Optional, Tuple
from data.language_environment import Language_Environment, Language_Observation, Policy
from craigslist.craigslist_base import Event, Role, Scene
from data.rl_data import Iterable_RL_Dataset, List_RL_Dataset, RL_Dataset
import random

class CraigslistObservation(Language_Observation):
    def __init__(self, scene: Scene, agent_role: Role, event: Optional[Event] = None):
        self.scene = scene
        self.agent_role = agent_role
        self.event = event

    def add(self, ev: Event):
        if self.event is not None:
            ev = self.event.append(ev)
        return CraigslistObservation(self.scene, self.agent_role, ev)

    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        sequence = [(self.scene.get_dialogue_header(), None)]
        if self.event is None:
            return sequence, False
        ev_rewards, _ = self.scene.get_rewards(self.scene, self.event, self.agent_role)
        sequence += [(str(ev), r) for ev, r in ev_rewards]
        return sequence, self.event.is_final()

    def __str__(self) -> str:
        dialogue_str = f'{self.scene.get_dialogue_header()}\n\n'
        if self.event is None:
            return dialogue_str.strip()
        for ev in self.event.get_events():
            dialogue_str += f'{str(ev)}\n'
        dialogue_str = dialogue_str.strip()
        return dialogue_str

    def metadata(self) -> Optional[Dict[str, Any]]:
        return {'scene': self.scene, 'event': self.event}
    
    def other(self):
        return CraigslistObservation(self.scene, self.agent_role.other(), self.event)

class CraigslistUserEnvironment(Language_Environment):
    def __init__(self, dataset: RL_Dataset, agent_role: Role):
        self.dataset = dataset
        self.agent_role = agent_role
        self.state = self.reset()

    def step(self, action: str) -> Tuple[CraigslistObservation, float, bool]:
        if self.state.event is not None and self.state.event.is_final():
            raise Exception("Cannot step after final action")
        self.state = self.state.add(Scene.parse_event(action, self.agent_role))
        print(self.state)
        if self.state.event is not None and self.state.event.is_final():
            _, reward = Scene.get_rewards(self.state.scene, self.state.event, self.agent_role)
            return self.state, reward, True
        print(f"Enter {self.agent_role.other()} Response:")
        response = input()
        response = f'{self.agent_role.other()}: ' + response
        self.state = self.state.add(Scene.parse_event(response, self.agent_role.other()))
        if self.state.event is not None and self.state.event.is_final():
            _, reward = Scene.get_rewards(self.state.scene, self.state.event, self.agent_role)
            return self.state, reward, True
        return self.state, 0.0, False

    def reset(self) -> CraigslistObservation:
        if isinstance(self.dataset, List_RL_Dataset):
            scene = self.dataset.get_item(random.choice(list(range(self.dataset.size())))).meta['scene']
        elif isinstance(self.dataset, Iterable_RL_Dataset):
            scene = self.dataset.sample_item().meta['scene']
        else:
            raise NotImplementedError
        self.state = CraigslistObservation(scene, self.agent_role)
        return self.state

    def is_terminal(self) -> bool:
        return self.state.event is not None and self.state.event.is_final()

class CraigslistPolicyEnvironment(Language_Environment):
    def __init__(self, response_policy: Policy, dataset: RL_Dataset, agent_role: Role, max_turns: Optional[int]):
        self.response_policy = response_policy
        self.dataset = dataset
        self.agent_role = agent_role
        self.max_turns = max_turns
        self.state = self.reset()

    def step(self, action: str) -> Tuple[CraigslistObservation, float, bool]:
        if self.state.event is not None and self.state.event.is_final():
            raise Exception("Cannot step after final action")
        self.state = self.state.add(Scene.parse_event(action, self.agent_role))
        if self.state.event is not None and self.state.event.is_final():
            _, reward = Scene.get_rewards(self.state.scene, self.state.event, self.agent_role)
            return self.state, reward, True
        response = self.response_policy.act(self.state.other())
        self.state = self.state.add(Scene.parse_event(response, self.agent_role.other()))
        if self.state.event is not None and (self.state.event.is_final() or (self.max_turns is not None and (len(self.state.event.get_events()) // 2) >= self.max_turns)):
            _, reward = Scene.get_rewards(self.state.scene, self.state.event, self.agent_role)
            return self.state, reward, True
        return self.state, 0.0, False

    def reset(self) -> CraigslistObservation:
        if isinstance(self.dataset, List_RL_Dataset):
            scene = self.dataset.get_item(random.choice(list(range(self.dataset.size())))).meta['scene']
        elif isinstance(self.dataset, Iterable_RL_Dataset):
            scene = self.dataset.sample_item().meta['scene']
        else:
            raise NotImplementedError
        self.state = CraigslistObservation(scene, self.agent_role)
        return self.state

    def is_terminal(self) -> bool:
        return self.state.event is not None and (self.state.event.is_final() or (self.max_turns is not None and (len(self.state.event.get_events()) // 2) >= self.max_turns))
