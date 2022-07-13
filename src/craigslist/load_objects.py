from craigslist.craigslist_base import CraigslistDialogueData
from craigslist.craigslist_dataset import CraigslistDataset
from craigslist.craigslist_env import CraigslistEnvironment
from load_objects import *
import pickle as pkl
from craigslist.craigslist_evaluator import TopAdvantageUtterances, Craigslist_Chai_Evaluator, Craigslist_DT_Evaluator, Craigslist_IQL_Evaluator, Utterance_Craigslist_IQL_Evaluator

@register('craigslist')
def load_craigslist(config, verbose=True):
    return CraigslistDialogueData(convert_path(config['data_path']),
                                  mode=config['mode'])

@register('craigslist_list_dataset')
def load_craigslist_list_dataset(config, device, verbose=True):
    craigslist = load_item(config['data'], verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return CraigslistDataset(craigslist, max_len=config['max_len'],
                                         token_reward=token_reward,
                                         top_p=config['top_p'],
                                         bottom_p=config['bottom_p'])

@register('craigslist_env')
def load_craigslist_env(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    return CraigslistEnvironment(dataset, config['url'],
                         actor_stop=config['actor_stop'])

@register('craigslist_remote_policy')
def load_craigslist_remote_policy(config, device, verbose=True):
    return VDRemotePolicy(config['url'])

@register('top_advantage_utterreward_cacheances_evaluator')
def load_top_advantage_utterances_evaluator(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    return TopAdvantageUtterances(data)

@register('craigslist_iql_evaluator')
def load_craigslist_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Craigslist_IQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('utterance_craigslist_iql_evaluator')
def load_utterance_craigslist_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Utterance_VisDial_IQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('craigslist_chai_evaluator')
def load_craigslist_chai_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return VisDial_Chai_Evaluator(env, config['verbose'], config['cache_save_path'], **config['generation_kwargs'])
