from craigslist.craigslist_base import CraigslistDialogueData
from data.rl_data import ConstantTokenReward
from craigslist.craigslist_dataset import CraigslistDataset

if __name__ == "__main__":
    d = CraigslistDataset(CraigslistDialogueData('../data/craigslist/test.json'), None, ConstantTokenReward(), None)
    print(d.size())