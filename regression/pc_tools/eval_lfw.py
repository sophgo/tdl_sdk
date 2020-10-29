import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', metavar='result', help='results txt path')
    args = parser.parse_args()

    result = [[0,0], [0,0], [0,0]]
    with open(args.result, 'r') as fp:
        content = fp.readlines()
        last_tpr = 0.0
        last_fpr = 0.0

        for line in content:
            tpr = float(line.split(',')[1][5:])
            fpr = float(line.split(',')[2][5:])

            if fpr <= 0.001:
                result[0][0] = tpr
                result[0][1] = fpr
            elif fpr <= 0.01:
                result[1][0] = tpr
                result[1][1] = fpr
            elif fpr <= 0.1:
                result[2][0] = tpr
                result[2][1] = fpr

    print('{}: '.format(args.result))
    print('[tpr,fpr]', result)
