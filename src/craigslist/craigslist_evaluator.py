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
from craigslist.craigslist_base import BuyerEvent, SellerEvent
from craigslist.craigslist_dataset import CraigslistDataset
from tqdm import tqdm
import torch
from collections import defaultdict
import numpy as np
import time
import random

class CraigslistBCGenerationEvaluator(Evaluator):
    def __init__(self, data: CraigslistDataset, kind: str, generation_kwargs: Dict[str, Any]):
        super().__init__()
        self.data = data
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
                if isinstance(ev, SellerEvent):
                    events = ev.get_events()
                    event = events[-2] if len(events) > 1 else None
                    history = CraigslistObservation(scene, event)
                    generation = policy.act(history)
                    print('='*25)
                    print(str(history))
                    print("="*25)
                    print('seller model:', generation)
                    print('='*25)
                    print()

# class Craigslist_IQL_Evaluator(Evaluator):
#     def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
#         super().__init__()
#         self.env = env
#         self.verbose = verbose
#         self.kind = kind
#         self.generation_kwargs = generation_kwargs
#         self.act_counts = []
#         self.all_results = []
#         # self.all_entropy = []
#         self.all_time = []
    
#     def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
#         policy = IQL_Policy(model, self.kind, **self.generation_kwargs)
#         tokens = model.prepare_inputs(items)['tokens']
#         total_token_reward = 0
#         total_env_reward = 0
#         total_activation_count = 0
#         for i in range(tokens.shape[0]):
#             s = time.time()
#             result, sequence = interact_environment(self.env, policy, None)
#             self.all_results.append((result, sequence,))
#             activation_count = sum(map(int, [self.env.yn_reward_f(ev.answer) if self.env.yn_reward_f is not None else 0 for ev in result.event.get_events() if isinstance(ev, AnswerEvent)]))
#             self.act_counts.append(activation_count / (len(result.event.get_events())/2))
#             env_reward = sum(map(lambda x: x[2], sequence))
#             token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
#             total_env_reward += env_reward
#             total_token_reward += token_reward
#             total_activation_count += activation_count
#             if self.verbose:
#                 print(result)
#                 print('='*25)
#                 print('token reward:', token_reward)
#                 print('env reward:', env_reward)
#                 print('activation count:', activation_count)
#                 print('avg token reward:', total_token_reward / (i + 1))
#                 print('avg env reward:', total_env_reward / (i + 1))
#                 print('avg activation count:', total_activation_count / (i + 1))
#                 print('='*25)
#             e = time.time()
#             self.all_time.append(e-s)
#         kl_total = sum(policy.kls_all)
#         time_total = sum(self.all_time)
#         print(np.histogram(self.act_counts))
#         return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 
#                 'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 
#                 'kl': (kl_total / len(policy.kls_all), len(policy.kls_all)), 
#                 'activation_count': (total_activation_count / tokens.shape[0], tokens.shape[0]), 
#                 'time': (time_total / len(self.all_time), len(self.all_time))}

#     def dump(self):
#         return {'results': self.all_results, 'histogram': self.act_counts, 'time': self.all_time}
