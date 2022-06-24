from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from typing import List, Optional
import json
import h5py
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod

N_TURNS = 10


def hard_yn_reward(text):
    text = text.lower()
    return text == 'yes' or text == 'no'


def soft_yn_reward(text):
    text = text.lower()
    return 'yes' in text or 'no' in text


def conservative_yn_reward(text):
    text = text.lower()
    key_words = ['not', 'don\'t', 'can\'t',
                 'don’t', 'can’t',
                 'cannot', 'fairly',
                 'could', 'think so',
                 'okay', 'maybe',
                 'yes', 'no',
                 'looks', 'appears',
                 'tell', 'mostly just']
    return any([word in text for word in key_words])


yn_reward_fs = {'none': None, 'soft': soft_yn_reward, 'hard': hard_yn_reward, 'conservative': conservative_yn_reward}

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
        return isinstance(self, StopEvent)


@dataclass
class BuyerEvent(Event):
    text: str
    action: str
    time: float
    reward: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return self.text


@dataclass
class SellerEvent(Event):
    text: str
    action: str
    time: float
    reward: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return self.text


@dataclass
class StopEvent(Event):
    progress: float
    scene: Scene
    prev: Optional[Event]
    next: Optional[Event]

    def __str__(self):
        return '<stop>'


@dataclass
class Scene:
    events: List[Event]
    initial_val: Optional[float]

    @classmethod
    def from_json(cls, scene_json, reward, progress):
        events = []
        for i in range(len(scene_json['events'][:-1])):
            if scene_json['events'][i]['agent'] == 0:
                events.append(BuyerEvent(scene_json['events'][i]['data'], scene_json['events'][i]['action'],
                                            scene_json['events'][i]['time'], 0.0,  None, None, None))
            else:
                events.append(SellerEvent(scene_json['events'][i]['data'], scene_json['events'][i]['action'],
                                            scene_json['events'][i]['time'], 0.0,  None, None, None))

        scene = cls(events, 0.0 if progress is None else progress[0])
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev = p
        for ev in events:
            ev.scene = scene
        return scene

    @classmethod
    def from_json_stops(cls, scene_json, reward, progress):
        events = []
        for i in range(len(scene_json['events'])):
            if scene_json['events'][i]['agent'] == 0:
                events.append(BuyerEvent(scene_json['events'][i]['data'], scene_json['events'][i]['action'],
                                            scene_json['events'][i]['time'], 0.0,  None, None, None))
            elif scene_json['events'][i]['agent'] == 1:
                events.append(SellerEvent(scene_json['events'][i]['data'], scene_json['events'][i]['action'],
                                            scene_json['events'][i]['time'], 0.0,  None, None, None))
            else:
                events.append(StopEvent(0.0, None, None, None))


        scene = cls(events, 0.0 if progress is None else progress[0])
        for p, n in zip(events[:-1], events[1:]):
            p.next = n
            n.prev = p
        for ev in events:
            ev.scene = scene
        return [scene]


class CraigslistDialogueData:
    def __init__(self, data_path: str,
                 reward_cache: Optional[str] = None,
                 reward_shift: float = 0.0,
                 reward_scale: float = 1.0,
                 mode: str = 'env_stops',
                 yn_reward: float = -2.0, yn_reward_kind: str = 'none'):
        assert mode in ['agent_stops']
        assert yn_reward_kind in yn_reward_fs
        yn_reward_f = yn_reward_fs[yn_reward_kind]
        with open(data_path, 'r') as f:
            data = json.load(f)
        if reward_cache is not None:
            with open(reward_cache, 'r') as f:
                reward = json.load(f)
            progress = reward
            reward = [[item * reward_scale + reward_shift for item in rs[1:]] for rs in reward]
        else:
            progress = None
            reward = None
        print(mode)
        if mode == 'agent_stops':
            self.scenes = sum([Scene.from_json_stops(data[i],
                                                     reward if reward is None else reward[i],
                                                     progress[i] if progress is not None else None) for i in range(len(data))], [])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]
