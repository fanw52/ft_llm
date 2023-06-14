import random

import jsonlines
import numpy as np

relation_dict = {"外_曾_祖父母": "祖父母或曾祖父母或外祖父母", "曾_孙子女": "孙子女或曾孙子女"}

all_relation = ['儿子', '兄弟姐妹', '兄弟姐妹的配偶', '其他亲属', '同事', '同窗', '喜欢', '祖父母或曾祖父母或外祖父母', '女儿', '子女',
                '子女的配偶', '师生', '影子实体', '恋爱', '孙子女或曾孙子女', '朋友', '未婚配偶', '母亲', '父亲', '父母', '社会关系',
                '离异配偶', '配偶', '配偶的父母']
all_relation_tail = ["兄弟姐妹的配偶", "其他亲属", "同事", "同窗", "喜欢", "子女的配偶", "孙子女或曾孙子女", "朋友", "未婚配偶", "父母",
                     "祖父母或曾祖父母或外祖父母", "离异配偶", "配偶的父母"]
# TODO: 修改use_sample，num_relaiotn_sample
use_sample = True
num_relaiotn_sample = 17
np.random.seed(0)
random.seed(0)
all_relation = sorted(all_relation)

# TODO：修改输入输出路径
path = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/relation_p2p_fine_grit_v2_v0.6/train_add_tail.jsonl"
output_path = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6/train_add_tail_do_sample.json"
# use_sample = False时保持
output_raw_path = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/relation_p2p_fine_grit_v2_v0.6/train_add_tail.json"


def generate_relation_data(path):
    a = 0
    result = []
    raw_result = []

    tmp = {}
    tmp1 = {}
    with jsonlines.open(path) as reader:
        for line in reader:
            content = line["content"]
            entity_list = line["entity_list"]
            output_str = ""

            relation_list = line["relation_list"]
            for relation in relation_list:
                v = relation["type"]
                v = relation_dict.get(v, v)
                if v not in tmp1:
                    tmp1[v] = 0
                tmp1[v] += 1

            if use_sample:
                # 每次采样包括70%的关系类型
                if len(relation_list) == 0:
                    continue

                # 如果当前数据包含长尾关系，那么这条长尾关系不能被丢弃
                candidate_relation = np.random.choice(all_relation, size=num_relaiotn_sample, replace=False, p=None)
                candidate_relation = list(candidate_relation)

                # 将没有采样到的长尾关系再添加回去
                for relation in relation_list:
                    v = relation["type"]
                    v = relation_dict.get(v, v)
                    if v in all_relation_tail and v not in candidate_relation:
                        candidate_relation.append(v)
                candidate_relation = sorted(candidate_relation)

            else:
                candidate_relation = all_relation

            candidate_relation_str = "，".join(candidate_relation)
            input_str = f"问题：下面句子包含{candidate_relation_str}关系的三元组是什么?\n句子:{content}"
            entity_dict = {}
            for entity in entity_list:
                entity_dict[entity['id']] = entity['value']

            relation_set = {}
            for relation in relation_list:
                src = entity_dict[relation['src']]
                dst = entity_dict[relation['dst']]
                v = relation["type"]
                v = relation_dict.get(v, v)

                if v not in candidate_relation:
                    continue

                if v not in relation_set:
                    relation_set[v] = []

                if (src, dst) not in relation_set[v]:
                    relation_set[v].append((src, dst))

                if v not in tmp:
                    tmp[v] = 0
                tmp[v] += 1

            for j, (k, v_list) in enumerate(relation_set.items()):
                output_str += f"具有{k}关系的头尾实体对如下:"
                for i, v in enumerate(v_list):
                    if i + 1 == len(v_list):
                        output_str += f"头实体为{v[0]}，尾实体为{v[1]}。"
                    else:
                        output_str += f"头实体为{v[0]}，尾实体为{v[1]}；"
                if j + 1 != len(relation_set.items()):
                    output_str += "\n"

            if len(output_str) == 0:
                output_str = "未找到上述关系的三元组"
                a += 1
            # if len(input_str)+len(output_str)>500:
            #     continue
            result.append({"input": input_str, "target": output_str})
            raw_result.append(line)
        print(f"未找到关系的数据共:{a}条")
        print(f"总数据量共:{len(result)}条")
        if use_sample:
            print("采样后的关系类型", sorted(tmp.items(), key=lambda item: item[0]))
            print("采样前的关系类型", sorted(tmp1.items(), key=lambda item: item[0]))
        return result, raw_result


def save2json(res, output_path):
    with jsonlines.open(output_path, 'w') as w:
        for line in res:
            w.write(line)


res, raw_res = generate_relation_data(path)
save2json(res, output_path)
if not use_sample:
    save2json(raw_res,output_raw_path)
print(len(res), len(raw_res))
