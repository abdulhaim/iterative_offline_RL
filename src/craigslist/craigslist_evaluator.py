from typing import Any, Dict, Optional
from craigslist.craigslist_env import CraigslistObservation
from models.base import Evaluator, InputType
from models.dt_model import DT_Policy
from models.iql_model import IQL_Policy, PerTokenIQL
from models.bc_lm import BC_LM, BC_Policy
from collections import defaultdict
from data.language_environment import Language_Environment, interact_environment
from data.rl_data import DataPoint
from models.utterance_iql_model import PerUtteranceIQL_Policy
from models.chai_model import ChaiModel, ChaiPolicy
from craigslist.craigslist_base import AcceptEvent, QuitEvent, RejectEvent, Role
from craigslist.craigslist_dataset import CraigslistDataset
from tqdm import tqdm
import torch
from collections import defaultdict
import numpy as np
import time
import random

class CraigslistBCGenerationEvaluator(Evaluator):
    def __init__(self, data: CraigslistDataset, agent_role: Role, kind: str, generation_kwargs: Dict[str, Any]):
        super().__init__()
        self.data = data
        self.agent_role = agent_role
        self.kind = kind
        self.generation_kwargs = generation_kwargs
    
    def evaluate(self, model: BC_LM, items: InputType) -> Optional[Dict[str, Any]]:
        policy = BC_Policy(model, self.kind, **self.generation_kwargs)
        items = model.prepare_inputs(items)
        n = items['tokens'].shape[0]
        for _ in range(n):
            datapoint = self.data.get_item(random.randint(0, self.data.size()-1))
            scene = datapoint.meta['scene']
            for ev in datapoint.meta['event'].get_events():
                if ev.role == self.agent_role:
                    events = ev.get_events()
                    event = events[-2] if len(events) > 1 else None
                    history = CraigslistObservation(scene, self.agent_role, event)
                    generation = policy.act(history)
                    print('='*25)
                    print(str(history))
                    print("="*25)
                    print(f'{self.agent_role} model:', generation)
                    print('='*25)
                    print()

def is_accept(obs: CraigslistObservation):
    ev = obs.event
    while ev is not None:
        if isinstance(ev, AcceptEvent):
            return True
        if isinstance(ev, RejectEvent) or isinstance(ev, QuitEvent):
            return False
        ev = ev.prev
    return False

class Craigslist_BC_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.all_results = []

    def evaluate(self, model: BC_LM, items: InputType) -> Optional[Dict[str, Any]]:
        policy = BC_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        n = tokens.shape[0]
        total_token_reward = 0
        total_env_reward = 0
        total_accept = 0
        for i in range(n):
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            accept = is_accept(result)
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_accept += float(accept)
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg accept:', total_accept / (i + 1))
                print('='*25)
        return {'token_reward': (total_token_reward / n, n), 'env_reward': (total_env_reward / n, n), 'accept_rate': (total_accept / n, n)}
    
    def dump(self):
        return {'all_results': self.all_results}

class Craigslist_IQL_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.all_results = []
        self.all_entropy = []
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        policy = IQL_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        total_token_reward = 0
        total_env_reward = 0
        total_accept = 0
        for i in range(tokens.shape[0]):
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            accept = is_accept(result)
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_accept += float(accept)
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg accept:', total_accept / (i + 1))
                print('='*25)
        kl_total = sum(policy.kls_all)
        entropy_total = -sum(policy.logprobs_all)
        self.all_entropy.extend(policy.logprobs_all)
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 'kl': (kl_total / len(policy.kls_all), len(policy.kls_all)), 
                'entropy': (entropy_total / len(policy.logprobs_all), len(policy.logprobs_all)), 'accept_rate': (total_accept / tokens.shape[0], tokens.shape[0])}
    
    def dump(self):
        return {'results': self.all_results, 'entropies': self.all_entropy}
