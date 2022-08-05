import argparse
import pickle as pkl
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    args = parser.parse_args()

    with open(args.eval_file, 'rb') as f:
        d = pkl.load(f)
    

    rs = [sum(map(lambda x: x[2], item[1])) for item in d['eval_dump']['results']]
    mean_r = np.mean(rs)
    std_r = np.std(rs)
    st_err_r = std_r / np.sqrt(len(rs))
    print(d['config'])
    print(f'reward: {mean_r} +- {st_err_r}')
    print('accept:', d['all_logs']['eval']['evaluation']['accept_rate'])
    print(list(d['eval_dump'].keys()))
    
    for item, sequence in d['eval_dump']['results']:
        print('='*25)
        print(item)
        print('='*25)
        print('reward:', sum(map(lambda x: x[2], sequence)))
        print('='*25)
        print()
        response = input('press "x" to stop, press any other key to continue: ')
        if response.lower() == "x":
            break
        print()
