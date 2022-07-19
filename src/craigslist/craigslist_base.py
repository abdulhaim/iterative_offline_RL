from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import json
import re

class Role(Enum):
    BUYER = "BUYER"
    SELLER = "SELLER"

    def other(self):
        if self == self.BUYER:
            return self.SELLER
        elif self == self.SELLER:
            return self.BUYER
        else:
            raise NotImplementedError

    def __str__(self):
        if self == self.BUYER:
            return "Buyer"
        elif self == self.SELLER:
            return "Seller"
        else:
            raise NotImplementedError

@dataclass
class Event:
    def append(self, ev: Event, link_forward=False):
        ev.prev = self
        if link_forward:
            self.next = ev
        ev.scene = self.scene
        return ev

    def get_events(self, direction="prev"):
        if direction == "prev":
            func = lambda ev: ev.prev
        elif direction == "next":
            func = lambda ev: ev.next
        else:
            raise NotImplementedError
        events = []
        ev = self
        while ev is not None:
            events.append(ev)
            ev = func(ev)
        if direction == 'prev':
            events.reverse()
        return events

    def get_all_events(self):
        return self.get_events() + self.get_events('next')[1:]

    def is_final(self):
        return isinstance(self, QuitEvent) or isinstance(self, RejectEvent) or isinstance(self, AcceptEvent)

@dataclass
class MessageEvent(Event):
    role: Role
    text: str
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return f'{self.role}: {self.text}'

@dataclass
class OfferEvent(Event):
    role: Role
    amount: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return f'{self.role}: OFFER {int(self.amount)}'

@dataclass
class AcceptEvent(Event):
    role: Role
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return f'{self.role}: ACCEPT'

@dataclass
class RejectEvent(Event):
    role: Role
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return f'{self.role}: REJECT'

@dataclass
class QuitEvent(Event):
    role: Role
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return f'{self.role}: QUIT'

@dataclass
class Scene:
    title: str
    description: str
    listing_price: float
    events: List[Event]

    @classmethod
    def from_json(cls, scene_json):
        events = []
        description = scene_json['scenario']['kbs'][0]['item']['Description'][0]
        listing_price = scene_json['scenario']['kbs'][0]['item']['Price']
        title =  scene_json['scenario']['kbs'][0]['item']['Title']
        for i in range(len(scene_json['events'])):
            if scene_json['events'][i]['agent'] == 0:
                if type(scene_json['events'][i]['data']) is dict:
                    if scene_json['events'][i]['action'] == 'offer':
                        if scene_json['events'][i]['data']['price'] is None:
                            continue
                        events.append(OfferEvent(Role.BUYER, scene_json['events'][i]['data']['price'], None, None, None))
                    else:
                        raise NotImplementedError
                elif scene_json['events'][i]['data'] is None:
                    if scene_json['events'][i]['action'] == 'accept':
                        events.append(AcceptEvent(Role.BUYER, None, None, None))
                    elif scene_json['events'][i]['action'] == 'reject':
                        events.append(RejectEvent(Role.BUYER, None, None, None))
                    elif scene_json['events'][i]['action'] == 'quit':
                        events.append(QuitEvent(Role.BUYER, None, None, None))
                    else:
                        raise NotImplementedError
                elif isinstance(scene_json['events'][i]['data'], str):
                    events.append(MessageEvent(Role.BUYER, scene_json['events'][i]['data'], None, None, None))
                else:
                    raise NotImplementedError
            elif scene_json['events'][i]['agent'] == 1:
                if type(scene_json['events'][i]['data']) is dict:
                    if scene_json['events'][i]['action'] == 'offer':
                        if scene_json['events'][i]['data']['price'] is None:
                            continue
                        events.append(OfferEvent(Role.SELLER, scene_json['events'][i]['data']['price'], None, None, None))
                    else:
                        raise NotImplementedError
                elif scene_json['events'][i]['data'] is None:
                    if scene_json['events'][i]['action'] == 'accept':
                        events.append(AcceptEvent(Role.SELLER, None, None, None))
                    elif scene_json['events'][i]['action'] == 'reject':
                        events.append(RejectEvent(Role.SELLER, None, None, None))
                    elif scene_json['events'][i]['action'] == 'quit':
                        events.append(QuitEvent(Role.SELLER, None, None, None))
                    else:
                        raise NotImplementedError
                elif isinstance(scene_json['events'][i]['data'], str):
                    events.append(MessageEvent(Role.SELLER, scene_json['events'][i]['data'], None, None, None))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        scene = cls(title, description, listing_price, events)
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev = p
        for ev in events:
            ev.scene = scene
        return scene
    
    def get_dialogue_header(self):
        return f'title: {self.title}\ndescription: {self.description}\nlisting_price: {self.listing_price}'
    
    @staticmethod
    def get_rewards(scene: Scene, last_event: Event, agent_role: Role) -> List[Tuple[Event, float]]:
        evs = last_event.get_events()
        reward = 0.0
        last_offer_idx = -1
        reward_ev = None
        for i, ev in enumerate(evs):
            if isinstance(ev, OfferEvent):
                reward = ev.amount / scene.listing_price
                last_offer_idx = i
        if last_offer_idx != -1:
            for ev in evs[(last_offer_idx+1):][::-1]:
                if ev.role == agent_role:
                    reward_ev = ev
                    break
        ev_rewards = []
        for ev in evs:
            if ev is reward_ev:
                ev_rewards.append((ev, reward,))
            else:
                if ev.role == agent_role:
                    ev_rewards.append((ev, 0.0,))
                else:
                    ev_rewards.append((ev, None,))
        return ev_rewards, reward

    @staticmethod
    def parse_event(event_str: str, expected_role: Role) -> Event:
        event_str = event_str.strip().lower()
        possible_events = [
            AcceptEvent(expected_role, None, None, None), 
            RejectEvent(expected_role, None, None, None), 
            QuitEvent(expected_role, None, None, None), 
        ]
        for ev in possible_events:
            if event_str == str(ev).strip().lower():
                return ev
        if expected_role == Role.BUYER:
            offer_re = r'^buyer: offer (\d+)$'
        elif expected_role == Role.SELLER:
            offer_re = r'^seller: offer (\d+)$'
        else:
            raise NotImplementedError
        re_match = re.match(offer_re, event_str)
        if re_match is not None:
            amount = float(re_match.group(1))
            return OfferEvent(expected_role, amount, None, None, None)
        if event_str.startswith(f'{str(expected_role).lower()}: '):
            return MessageEvent(expected_role, event_str[len(f'{str(expected_role).lower()}: '):], None, None, None)
        return MessageEvent(expected_role, event_str, None, None, None)

class CraigslistDialogueData:
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.scenes = list(filter(lambda x: len(x.events) > 0, [Scene.from_json(data[i]) for i in range(len(data))]))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]
