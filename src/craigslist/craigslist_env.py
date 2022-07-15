from typing import Any, Dict, Iterator, List, Optional, Tuple
from data.language_environment import Language_Environment, Language_Observation, Policy
from craigslist.craigslist_base import SellerEvent, Event, Scene


class CraigslistObservation(Language_Observation):
    def __init__(self, scene: Scene, event: Optional[Event] = None):
        self.scene = scene
        self.event = event

    def add(self, ev: Event):
        if self.event is not None:
            ev = self.event.append(ev)
        return CraigslistObservation(self.scene, ev)

    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.event is None:
            return [(self.scene.description, None)], False
        evs = self.event.get_events()
        sequence = [(self.scene.description, None)]
        sequence += [(str(evs[i]), 0.0 if isinstance(evs[i], SellerEvent) else None) for i in range(len(evs))]
        # terminal = self.event.is_final()
        # Temporary: make terminal always tue, change this later when we want to do automatic evaluations.
        terminal = True
        return sequence, terminal

    def __str__(self) -> str:
        dialogue_str = f'description: {self.scene.description}\n\n'
        if self.event is None:
            return dialogue_str.strip()
        for ev in self.event.get_events():
            speaker = 'seller' if isinstance(ev, SellerEvent) else 'buyer'
            dialogue_str += f'{speaker}: {str(ev)}\n'
        dialogue_str = dialogue_str.strip()
        return dialogue_str

    def metadata(self) -> Optional[Dict[str, Any]]:
        return {'scene': self.scene, 'event': self.event}


# class CraigslistEnvironment(Language_Environment):
#     def __init__(self, dataset: RL_Dataset, url: str, actor_stop: bool = False):
#         self.dataset = dataset
#         self.state = self.reset()
#         self.actor_stop = actor_stop

#     def step(self, action: str) -> Tuple[CraigslistObservation, float, bool]:
#         print("Enter Buyer Response:")
#         buyer_input = input()
#         print(action)

#         if self.state.event is not None and self.state.event.is_final():
#             raise Exception("Cannot step after final action")
#         if self.state.event is not None and self.state.event.is_final():
#             reward = self.state.event.final_reward
#             self.state = self.state.add(StopEvent(reward, None, None, None))
#         else:
#             from datetime import datetime
#             reward = 0.0
#             self.state = self.state.add(BuyerEvent(action, "statement", str(datetime.now()), reward, None, None, None))

#         return self.state, reward, False

#     def reset(self) -> CraigslistObservation:
#         if isinstance(self.dataset, List_RL_Dataset):
#             scene = self.dataset.get_item(random.choice(list(range(self.dataset.size())))).meta['scene']
#         elif isinstance(self.dataset, Iterable_RL_Dataset):
#             scene = self.dataset.sample_item().meta['scene']
#         else:
#             raise NotImplementedError
#         self.state = CraigslistObservation(scene)
#         return self.state

#     def is_terminal(self) -> bool:
#         return self.state.event is not None and self.state.event.is_final()

#
# class CraigslistRemotePolicy(Policy):
#     def __init__(self, url: str) -> None:
#         self.url = url
#
#     def act(self, obs: CraigslistObservation):
#         history = []
#         if obs.event is not None:
#             for item in obs.event.get_events():
#                 if isinstance(item, BuyerEvent):
#                     history.append({'speaker': 'buyer', 'text': item.text})
#                 elif isinstance(item, AnswerEvent):
#                     history.append({'speaker': 'seller', 'text': item.text})
#                 else:
#                     raise NotImplementedError
#         payload = {'history': json.dumps(history),
#                    'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}
#         q_response = json.loads(requests.post(self.url, data=payload).text)
#         return q_response


# class CraigslistRemoteReward:
#     def __init__(self, url: str) -> None:
#         self.url = url

#     def reward(self, obs: CraigslistObservation):
#         history = []
#         if obs.event is not None:
#             for item in obs.event.get_events():
#                 if isinstance(item, BuyerEvent):
#                     history.append({'buyer': item.text})
#                 elif isinstance(item, SellerEvent):
#                     history.append({'seller': item.text})
#                 else:
#                     raise NotImplementedError
#         payload = {'history': json.dumps(history)}
#         q_response = json.loads(requests.post(self.url,
#                                               data=payload).text)
#         return q_response
