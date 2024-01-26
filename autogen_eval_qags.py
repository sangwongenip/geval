from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau

import json
import re
import argparse


def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result

def extract_numbers(text):
    # 정규 표현식을 사용하여 텍스트에서 숫자를 찾습니다.
    numbers = re.findall(r'\d+', text)
    # 찾은 숫자들을 정수형 리스트로 변환합니다.
    return [int(num) for num in numbers]


def print_correlations(result, n):
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if (n == 0):
        n = 1
    table.add_row(
        [round(result['pearson'] / n, 4), round(result['spearman'] / n, 4), round(result['kendalltau'] / n, 4)])
    print(table)


def parse_output(output):
    matched = re.search("^ ?([\d\.]+)", output)
    if (matched):
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        score = 0
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fp', type=str, default='results/gpt4_con_detailed_openai.json')
    parser.add_argument('--human_data', type=str, default='/Users/keonwookim/anaconda3/envs/icml/workspace/geval/results/gpt4_rel_detailed.json')
    parser.add_argument('--dimension', type=str, default='consistency')
    parser.add_argument('--summeval_fp', type=str, default='data/summeval.json')
    args = parser.parse_args()

    summeval = json.load(open(args.summeval_fp))
    jobj = json.load(open(args.input_fp))
    human_j = json.load(open(args.human_data))
    pred_scores, human_scores = {}, {}

    print("Calculating correlation for G-Eval")
    for item in jobj:
        doc_id = item["doc_id"]
        if (doc_id not in pred_scores):
            pred_scores[doc_id] = []
            human_scores[doc_id] = []

        response = item[args.dimension]
        score = int(extract_numbers(response)[0])


        pred_scores[doc_id].append(score)

        human_score = item['human_score']
        human_scores[doc_id].append(human_score)

    print('len(pred_scores): {}'.format(len(pred_scores)))
    print('len(human_scores): {}'.format(len(human_scores)))

    results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    d_ctr = 0
    for doc_id in pred_scores:
        pred_scores_doc = pred_scores[doc_id]
        human_scores_doc = human_scores[doc_id]
        if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
            continue

        results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
