import pickle


def make_pickle_file(dataset):
    assert dataset in ['dailydialog', 'multiwoz']
    raw_path = {s: make_raw_path(dataset, s) for s in ['train', 'val', 'test']}
    save_path = {s: make_save_path(dataset, s) for s in ['train', 'val', 'test']}

    if dataset == 'dailydialog':
        for s, p in raw_path.items():
            with open(p, 'rb') as f:
                data = pickle.load(f)

            tmp = [tuple([s for s in d['utterances']]) for d in data]
            with open(save_path[s], 'wb') as f:
                pickle.dump(tmp, f)


def make_raw_path(dataset, state):
    return './data/' + dataset + '/raw/' + dataset + '.' + state


def make_save_path(dataset, state):
    return './data/' + dataset + '/processed/' + dataset + '.' + state


if __name__ == '__main__':
    dataset = 'dailydialog'
    make_pickle_file(dataset)
