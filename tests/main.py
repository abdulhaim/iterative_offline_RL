from craigslist.craigslist_base import CraigslistDialogueData, Role
from data.rl_data import ConstantTokenReward
from craigslist.craigslist_dataset import CraigslistDataset
import numpy as np

if __name__ == "__main__":
    # d = CraigslistDataset(CraigslistDialogueData('../data/craigslist/train.json'), Role.SELLER, 983, ConstantTokenReward(), None)
    # print(d.get_item(0).raw_str)
    # min_r = float('inf')
    # max_r = float('-inf')
    # for i in range(d.size()):
    #     rs = np.array(list(filter(lambda x: x is not None, [r for _, r in d.get_item(i).meta['scene'].get_rewards(d.get_item(i).meta['scene'], d.get_item(i).meta['event'], Role.SELLER)[0]])))
    #     if np.any(np.isnan(rs)):
    #         print(rs)
    #     if np.max(rs) < 0:
    #         print(str(d.get_item(i)))
        # print(d.get_item(i).meta['scene'].get_dialogue_header() + '\n'.join(map(lambda x: str(x), d.get_item(i).meta['scene'].events)))
        # print(rs)
        # print()
    #     min_r, max_r = min(min_r, np.min(rs)), max(max_r, np.max(rs))
    # print(min_r, max_r)
    pass
