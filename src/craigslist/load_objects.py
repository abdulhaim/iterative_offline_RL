from craigslist.craigslist_base import CraigslistDialogueData, Role
from craigslist.craigslist_dataset import CraigslistDataset
from craigslist.craigslist_env import CraigslistPolicyEnvironment, CraigslistUserEnvironment
from craigslist.craigslist_evaluator import Craigslist_IQL_Evaluator, Craigslist_BC_Evaluator, CraigslistBCGenerationEvaluator
# from craigslist.craigslist_env import CraigslistEnvironment
from load_objects import *
import pickle as pkl
# from craigslist.craigslist_evaluator import TopAdvantageUtterances, Craigslist_Chai_Evaluator, Craigslist_DT_Evaluator, Craigslist_IQL_Evaluator, Utterance_Craigslist_IQL_Evaluator

def get_agent_role(role_str):
    if role_str == 'buyer':
        return Role.BUYER
    elif role_str == 'seller':
        return Role.SELLER
    else:
        raise NotImplementedError

@register('craigslist')
def load_craigslist(config, verbose=True):
    return CraigslistDialogueData(convert_path(config['data_path']))

@register('craigslist_list_dataset')
def load_craigslist_list_dataset(config, device, verbose=True):
    craigslist = load_item(config['data'], verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return CraigslistDataset(craigslist, 
                             get_agent_role(config['agent_role']), 
                             max_len=config['max_len'], 
                             token_reward=token_reward, 
                             reward_mode=config['reward_mode'], 
                             top_p=config['top_p'])

@register('craigslist_bc_generation_evaluator')
def load_craigslist_bc_generation_evaluator(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    return CraigslistBCGenerationEvaluator(data, get_agent_role(config['agent_role']), 
                                           config['kind'], config['generation_kwargs'])

@register('craigslist_user_env')
def load_craigslist_user_env(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    return CraigslistUserEnvironment(dataset, get_agent_role(config['agent_role']), reward_mode=config['reward_mode'])

@register('craigslist_policy_env')
def load_craigslist_policy_env(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    policy = load_item(config['policy'], device, verbose=verbose)
    return CraigslistPolicyEnvironment(policy, dataset, get_agent_role(config['agent_role']), reward_mode=config['reward_mode'], max_turns=config['max_turns'])

@register('craigslist_bc_evaluator')
def load_craigslist_bc_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Craigslist_BC_Evaluator(env=env, verbose=config['verbose'], kind=config['kind'], **config['generation_kwargs'])

@register('craigslist_iql_evaluator')
def load_craigslist_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Craigslist_IQL_Evaluator(env=env, verbose=config['verbose'], kind=config['kind'], **config['generation_kwargs'])
