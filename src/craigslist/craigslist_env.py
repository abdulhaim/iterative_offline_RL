from typing import Any, Dict, Iterator, List, Optional, Tuple
from data.language_environment import Language_Environment, Language_Observation, Policy
import requests
import json
from data.rl_data import List_RL_Dataset, Iterable_RL_Dataset, RL_Dataset
import random
from craigslist.craigslist_base import BuyerEvent, SellerEvent, Event, Scene, N_TURNS, StopEvent
from craigslist.craigslist_base import yn_reward_fs


class CraigslistObservation(Language_Observation):
    def __init__(self, scene: Scene, event: Optional[Event] = None):
        self.scene = scene
        self.event = event

    def add(self, ev: Optional[Event]):
        if self.event is not None:
            ev = self.event.append(ev)
        elif ev is not None:
            ev.scene = self.scene
        return CraigslistObservation(self.scene, ev)

    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.event is None:
            return [(self.scene.caption, None)], False
        evs = self.event[0].get_events()
        sequence = []
        sequence += [(str(evs[i]), evs[i + 1].reward if isinstance(evs[i + 1], BuyerEvent) else None) for i in
                     range(len(evs) - 1)]
        sequence += [(str(evs[i]), evs[i + 1].reward if isinstance(evs[i + 1], SellerEvent) else None) for i in
                     range(len(evs) - 1)]
        sequence += [(str("offer"), 0.0 if isinstance(evs[-1], StopEvent) else None)]
        terminal = self.event[0].is_final()
        return sequence, terminal

    def __str__(self) -> str:
        return list(map(str, self.event.get_events()))

    def metadata(self) -> Optional[Dict[str, Any]]:
        return {'scene': self.scene, 'event': self.event}


class CraigslistEnvironment(Language_Environment):
    def __init__(self, dataset: RL_Dataset, url: str, reward_shift: float = 0.0,
                 reward_scale: float = 1.0, actor_stop: bool = False, yn_reward: float = -2.0,
                 yn_reward_kind: str = 'none'):
        self.dataset = dataset
        self.remote_env = CraigslistEnvRemoteWrapper(url)
        self.state = self.reset()
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.actor_stop = actor_stop
        self.yn_reward = yn_reward
        self.yn_reward_f = yn_reward_fs[yn_reward_kind]

    def step(self, action: str) -> Tuple[CraigslistObservation, float, bool]:
        if self.state.event is not None and self.state.event.is_final():
            raise Exception("Cannot step after final action")
        if '<stop>' in action and self.actor_stop:
            self.state = self.state.add(StopEvent(0.0, None, None, None))
            reward = 0.0 * self.reward_scale + self.reward_shift
        else:
            self.state = self.state.add(QuestionEvent(action, 0.0, None, None, None))
            response, reward = self.remote_env.step(self.state)
            progress = reward
            reward = reward * self.reward_scale + self.reward_shift
            # else:
            #     reward = (-1.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(
            #         response) else 0.0)) * self.reward_scale + self.reward_shift
            self.state = self.state.add(AnswerEvent(response, reward, progress, None, None, None))
            if self.state.event.is_final():
                self.state.event.reward = (0.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(
                    response) else 0.0)) * self.reward_scale + self.reward_shift
                reward = (0.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(
                    response) else 0.0)) * self.reward_scale + self.reward_shift
        return self.state, reward, self.state.event.is_final()

    def reset(self) -> CraigslistObservation:
        if isinstance(self.dataset, List_RL_Dataset):
            scene = self.dataset.get_item(random.choice(list(range(self.dataset.size())))).meta['scene']
        elif isinstance(self.dataset, Iterable_RL_Dataset):
            scene = self.dataset.sample_item().meta['scene']
        else:
            raise NotImplementedError
        self.state = CraigslistObservation(scene)
        return self.state

    def is_terminal(self) -> bool:
        return self.state.event is not None and self.state.event.is_final()


class CraigslistEnvRemoteWrapper:
    def __init__(self, url: str) -> None:
        self.url = url

    def step(self, obs: CraigslistObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, BuyerEvent):
                    history.append({'speaker': 'buyer', 'text': item.text})
                elif isinstance(item, SellerEvent):
                    history.append({'speaker': 'seler', 'text': item.text})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history),
                   'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}
        a_response, reward = json.loads(requests.post(self.url,
                                                      data=payload).text)
        return a_response, reward


class CraigslistRemotePolicy(Policy):
    def __init__(self, url: str) -> None:
        self.url = url

    def act(self, obs: CraigslistObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, BuyerEvent):
                    history.append({'speaker': 'buyer', 'text': item.text})
                elif isinstance(item, AnswerEvent):
                    history.append({'speaker': 'seller', 'text': item.text})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history),
                   'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}
        q_response = json.loads(requests.post(self.url,
                                              data=payload).text)
        return q_response


class CraigslistRemoteReward:
    def __init__(self, url: str) -> None:
        self.url = url

    def reward(self, obs: CraigslistObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, BuyerEvent):
                    history.append({'buyer': item.text})
                elif isinstance(item, SellerEvent):
                    history.append({'seller': item.text})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history)}
        q_response = json.loads(requests.post(self.url,
                                              data=payload).text)
        return q_response
