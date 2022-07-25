from datasets import list_datasets, load_dataset
from pprint import pprint
import pickle


def download(state='show', split=None, path=None):
    if state == 'show':
        pprint(list_datasets(), compact=True)
    else:
        if split == None:
            dataset = load_dataset(state)
            print('Please select split...\n', dataset)
        else:
            dataset = load_dataset(state, split=split)
            with open(path, 'wb') as f:
                pickle.dump(dataset, f)
            print('{} of {} data are saved'.format(len(dataset), split))


if __name__ == '__main__':
    data_name = 'show'
    split = 'test'
    path = './data/dailydialog/raw/dailydialog.test'

    download(data_name, split, path)
