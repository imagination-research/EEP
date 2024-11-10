import math
import json
import argparse
from typing import List
from sklearn.metrics import accuracy_score, auc


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def main(args):
    labels_file = args.labels_file
    preds_file = args.preds_file
    metrics_output_file = args.metrics_output_file

    gold_answers = [l.strip() for l in open(labels_file, 'r')]
    pred_answers_list = [l.strip().split(',') for l in open(preds_file, 'r')]
    pred_answers_list = list(zip(*pred_answers_list))

    training_split = ['XS', 'S', 'M', 'L', 'XL']
    training_sizes = [160, 640, 2558, 10234, 40398]
    x = [math.log2(t) for t in training_sizes]
    x_diff = max(x)-min(x)

    results = {}

    y = []
    for train_name, pred_answers in zip(training_split, pred_answers_list):
        if len(gold_answers) != len(pred_answers):
            raise Exception("The prediction file seems incomplete or formated incorrectly.")

        accuracy = accuracy_score(gold_answers, pred_answers)
        results['accuracy_' + train_name] = accuracy
        y.append(accuracy)

    results["AUC"] = auc(x, y)/x_diff       # normalized area under (learing) curve

    with open(metrics_output_file, "w") as f:
        f.write(json.dumps(results))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate WinoGrande predictions')
    # Required Parameters
    parser.add_argument('--labels_file', type=str, help='Location of test labels', default=None)
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default=None)
    parser.add_argument('--metrics_output_file',
                        type=str,
                        help='Location of output metrics file',
                        default="metrics.json")

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
