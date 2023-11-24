import jsonlines

if __name__ == '__main__':
    eval_file = "/data/wqx/llm-experiments/data/jq_classify_test_data/valid_10.jsonl"
    all_num_tp, all_num_fp, all_num_fn = 0, 0, 0
    with jsonlines.open(eval_file) as reader:
        for line in reader:
            answer = line.get("answer", "")
            target = line["target"]
            targegt_set = set(target.split(","))
            answer_set = set(answer.split(","))
            tp = targegt_set.intersection(answer_set)
            fp = targegt_set.difference(answer_set)
            fn = targegt_set.difference(answer_set)
            num_fn = len(fn)
            num_fp = len(fp)
            all_num_tp += len(tp)
            all_num_fn += len(fn)
            all_num_fp += len(fp)

    print(all_num_tp, all_num_fn, all_num_fp)
    precision = all_num_tp / (all_num_tp + all_num_fp)
    recall = all_num_tp / (all_num_tp + all_num_fn)
    f1 = 2 * precision * recall / (precision + recall)
    info = f"precision:{precision:.04f}\nrecall:{recall:.04f}\nf1:{f1:.04f}"
    print(info)
