import random
from tqdm import tqdm
from argparse import ArgumentParser


def pp(args):
    random.seed(999)
    read_path = args.input_path
    write_path = args.output_path

    with open(read_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    line_fin = []
    for line in tqdm(lines):
        try:
            if line[-1] in ['.', '?', '!'] and len(line) > 10 and line[0] not in ['.', '?', '!', ',']:
                line_fin.append(line)
        except IndexError:
            continue
    try:
        line_fin = random.sample(list(set(line_fin)), args.num)
    except:
        line_fin = random.sample(list(set(line_fin)), len(line_fin))

    line_fin = [line + '\n' for line in line_fin]
    with open(write_path, 'w') as f:
        f.writelines(line_fin)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('-n', '--num', default=50, type=int)
    args = parser.parse_args()

    pp(args)


