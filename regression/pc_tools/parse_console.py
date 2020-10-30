import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', metavar='log', help='Console log path')
    args = parser.parse_args()

    with open(args.log, 'r') as fp:
        content = fp.readlines()

        for idx, line in enumerate(content):
            if line.find('db count') != -1:
                print('Face attribute ZKT:')
                print(content[idx], content[idx+1], content[idx+2])
            elif line.find('pos num') != -1:
                print('Face quality:')
                print(content[idx], content[idx+1])
            elif line.find('Num of mask face') != -1:
                print('Mask classification:')
                print(content[idx])
            elif line.find('Rank [1]') != -1:
                print('Reid:')
                print(content[idx], content[idx+1])
            elif line.find('total distance') != -1:
                print('Face align:')
                print(content[idx])
            elif line.find('Num of esc') != -1:
                print('ES classification:')
                print(content[idx])
