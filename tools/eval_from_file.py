import jieba
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import  jsonlines

def compute_metrics(decoded_preds, decoded_labels ):

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


if __name__ == '__main__':
    path = "/data/wufan/data/wx_bilu_aug/val_aug_0531_pred.json"
    decoded_preds = []
    decoded_labels = []
    with jsonlines.open(path) as reader:
        for line in reader:
            target = line["target"]
            answer = line["answer"]
            decoded_preds.append(answer)
            decoded_labels.append(target)

    score_dict = compute_metrics(decoded_preds=decoded_preds,decoded_labels=decoded_labels)
    print(score_dict)