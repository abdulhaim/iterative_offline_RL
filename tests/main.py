from craigslist.craigslist_base import CraigslistDialogueData, Role
from data.rl_data import ConstantTokenReward
from craigslist.craigslist_dataset import CraigslistDataset

if __name__ == "__main__":
    d = CraigslistDataset(CraigslistDialogueData('../data/craigslist/test.json'), Role.BUYER, None, ConstantTokenReward(), None)
    # print(d.get_item(0).raw_str)