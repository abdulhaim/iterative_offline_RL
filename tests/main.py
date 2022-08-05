from craigslist.craigslist_base import CraigslistDialogueData, Role
from data.rl_data import ConstantTokenReward
from craigslist.craigslist_dataset import CraigslistDataset
import numpy as np

if __name__ == "__main__":
    """This apartment building provides you with an excellent location! It is only two blocks from the UC Berkeley campus and just a few blocks from the Downtown Berkeley BART station and Telegraph Avenue! This unit provides a brand new modern kitchen, abundant natural lighting for comfortable living, and is well-maintained and organized with a modern and relaxing appeal. This unit is unfurnished."""
    d = CraigslistDataset(CraigslistDialogueData('../data/craigslist/eval.json'), Role.SELLER, 983, ConstantTokenReward(), 'agent_utility', None)
    for i in range(d.size()):
        if 'two blocks from the UC Berkeley campus' in d.get_item(i).raw_str:
            print(d.get_item(i).raw_str)
    # min_r = float('inf')
    # max_r = float('-inf')
    # for i in range(d.size()):
    #     print(d.get_item(i).meta['scene'].get_dialogue_header() + '\n'.join(map(lambda x: str(x), d.get_item(i).meta['scene'].events)))
    #     print(d.get_item(i).rewards)
        # rs = np.array(d[i].rewards)
        # print(rs)
        # if np.any(np.isnan(rs)):
        #     print(rs)
        # if np.max(rs) < 0:
        #     print(str(d.get_item(i)))
        # print(d.get_item(i).meta['scene'].get_dialogue_header() + '\n'.join(map(lambda x: str(x), d.get_item(i).meta['scene'].events)))
        # print(rs)
        # print()
        # min_r, max_r = min(min_r, np.min(rs)), max(max_r, np.max(rs))
    # print(min_r, max_r)
