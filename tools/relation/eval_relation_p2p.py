import re

import jsonlines

unorder_relation_list = ["兄弟姐妹", "其他亲属", "同事", "同窗", "喜欢", "恋爱", "朋友",
                         "未婚配偶", "社会关系", "离异配偶", "配偶", "师生", "影子实体"]




def process_relation_pair(pair, relation):
    if relation in unorder_relation_list:
        pair = sorted(pair)
    return (pair[0], relation, pair[1])


def text2structure(text, special="未找到上述关系的三元组"):
    pattern = "具有(.{1,20})关系的头尾实体对如下"
    src_pattern = "头实体为(.{1,20})[,，]"
    dst_pattern = "尾实体为(.{1,20})。"

    structure = set()
    if special != text:
        text_list = text.split("\n")
        pattern = re.compile(pattern)
        src_pattern = re.compile(src_pattern)
        dst_pattern = re.compile(dst_pattern)

        for ss in text_list:
            relation_list = pattern.findall(ss)
            if len(relation_list):
                relation = relation_list[0]

                ss_list = ss.split(";")
                for s in ss_list:
                    if not s.endswith("。"):
                        s_new = s + "。"
                    else:
                        s_new = s
                    src = src_pattern.findall(s_new)
                    dst = dst_pattern.findall(s_new)

                    if len(src) and len(dst):
                        if src == dst:
                            continue
                        pair = process_relation_pair(pair=(src[0], dst[0]), relation=relation)
                        structure.add(pair)
    return structure


if __name__ == '__main__':

    eval_file = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6/valid_chatglm.json"

    all_num_tp, all_num_fp, all_num_fn = 0, 0, 0
    with jsonlines.open(eval_file) as reader:
        for line in reader:
            answer = line["answer"]
            target = line["target"]
            pairs_answer = text2structure(answer)
            pairs_target = text2structure(target)

            tp = pairs_target.intersection(pairs_answer)
            fp = pairs_answer.difference(pairs_target)
            fn = pairs_target.difference(pairs_answer)
            num_fn = len(fn)
            num_fp = len(fp)
            if num_fp != 0 or num_fp != 0:
                print(line["input"])
                print("pairs_target", pairs_target)
                print("pairs_answer", pairs_answer)
                print("fn", fn)
                print("fp", fp)
                print()
            all_num_tp += len(tp)
            all_num_fn += len(fn)
            all_num_fp += len(fp)

    print(all_num_tp, all_num_fn, all_num_fp)
    precision = all_num_tp / (all_num_tp + all_num_fp)
    recall = all_num_tp / (all_num_tp + all_num_fn)
    f1 = 2 * precision * recall / (precision + recall)
    info = f"precision:{precision:.04f}\nrecall:{recall:.04f}\nf1:{f1:.04f}"
    print(info)
