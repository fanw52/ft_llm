import os

import jsonlines
import pandas as pd


in_data_dir = "/data/algo/model/models_train_kp/jq-classify-server/jq_classify_v7.2.18-202310复现-新处理数据/data/preprocessed_data"
out_data_dir = "/data/wufan/llm-experiments/data/jq_classify"

if not os.path.exists(out_data_dir):
    os.makedirs(out_data_dir)

filename_list = ["train.jsonl", "valid.jsonl", "blind.jsonl"]

label_path = "labels_map_v2.xlsx"
label_lines = pd.read_excel(label_path)
label_lines = label_lines.to_records()

label_dict = {}
for l in label_lines:
    label_dict[l[1]] = l[2]


for filename in filename_list:
    in_path = os.path.join(in_data_dir, filename)
    result = []

    with jsonlines.open(in_path) as reader:
        for line in reader:
            content = line["content"]
            labels = line["labels"]
            labels = [label_dict.get(l, l) for l in labels]
            result.append({"instruction": instruction, "input": content, "target": "\n".join(labels)})

    out_path = os.path.join(out_data_dir, filename.split(".")[0] + ".json")

    with jsonlines.open(out_path, 'w') as w:
        for line in result:
            w.write(line)



#
# instruction = '''###instruction:针对下面的报警或处警内容，判断该案件内容所属的案件类别，案件类别包括:{label_set}\n\n###input'''
# base_model = "/data/pretrained_models/Baichuan2-13B-Chat"
#
# labels_path = "/data/wqx/llm-experiments/data/jq_classify_test_data/labels_map_v2.xlsx"
# import pandas as pd
#
# data = pd.read_excel(labels_path)
# data_list = data.to_records()
#
# label_list = []
# for line in data_list:
#     print(line[3])
#     label_list.append(line[3])
#
# s = f",".join(label_list)
# print(s)
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)
#
# s= tokenizer.encode(instruction.format_map({"label_set":s}))
# print(len(s))
