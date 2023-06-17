import jsonlines

all_cls_list = ['中毒事故', '交通事故', '交通逃逸', '传销', '伤害、殴打他人', '伤害致死', '其他违法行为', '制作传播贩卖淫秽', '劫持', '劳动生产事故', '卖淫嫖娼',
                '危化物品泄漏事故', '吸毒', '寻衅滋事', '建筑坍塌事故', '强奸猥亵', '意外伤亡', '投放危险物质', '抢劫抢夺', '拐卖', '挤压死伤事故', '故意伤害', '敲诈勒索',
                '斗殴', '杀人', '求助', '涉外事件', '涉恐', '涉枪', '涉邪教', '火灾', '灾害事故', '爆炸爆燃事故', '疫情灾害', '盗窃', '纵火', '组织卖淫', '绑架',
                '群众求助', '自杀', '自然天气灾害事故', '诈骗', '贩毒', '赌博', '走私', '走私毒品', '辟谣', '阻碍执行职务', '非法侵入住宅', '非法制售爆炸物品', '非法猎捕',
                '非法盗伐林木', '非法限制人身自由']


def generate_jqtb_cls_data(path):
    result = []
    with jsonlines.open(path) as reader:
        for line in reader:
            content = line["text"]
            labels = line["labels"]
            cls_str = "，".join(all_cls_list)
            input_str = f"问题：下面句子包含的事件类型，选项：{cls_str}\n句子:{content}"
            if len(labels):
                target = f"上述句子中包含的事件类型包括:{'，'.join(labels)}"
            else:
                target = "上述句子中不包含选项中的事件类型。"
            result.append({"input": input_str, "target": target})

        return result


def save2json(res, output_path):
    with jsonlines.open(output_path, 'w') as w:
        for line in res:
            w.write(line)


if __name__ == '__main__':

    # TODO：修改输入输出路径
    path = "/data/wufan/data/jqtb_cls/jqtb_1.0.12/data/jqtb_classify/train.jsonl"
    output_path = "/data/wufan/data/jqtb_cls/valid.json"

    res = generate_jqtb_cls_data(path)
    # save2json(res, output_path)
    print(len(res))
