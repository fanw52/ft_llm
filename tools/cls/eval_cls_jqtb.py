import re

import jsonlines

all_cls_list = ['中毒事故', '交通事故', '交通逃逸', '传销', '伤害、殴打他人', '伤害致死', '其他违法行为', '制作传播贩卖淫秽', '劫持', '劳动生产事故', '卖淫嫖娼',
                '危化物品泄漏事故', '吸毒', '寻衅滋事', '建筑坍塌事故', '强奸猥亵', '意外伤亡', '投放危险物质', '抢劫抢夺', '拐卖', '挤压死伤事故', '故意伤害', '敲诈勒索',
                '斗殴', '杀人', '求助', '涉外事件', '涉恐', '涉枪', '涉邪教', '火灾', '灾害事故', '爆炸爆燃事故', '疫情灾害', '盗窃', '纵火', '组织卖淫', '绑架',
                '群众求助', '自杀', '自然天气灾害事故', '诈骗', '贩毒', '赌博', '走私', '走私毒品', '辟谣', '阻碍执行职务', '非法侵入住宅', '非法制售爆炸物品', '非法猎捕',
                '非法盗伐林木', '非法限制人身自由']


def text2structure(text, special="上述句子中不包含选项中的事件类型。"):
    '''
    问题:下面句子包含的事件类型,选项:中毒事故,交通事故,交通逃逸,传销,伤害、殴打他人,伤害致死,其他违法行为,制作传播贩卖淫秽,劫持,劳动生产事故,卖淫嫖娼
    ,危化物品泄漏事故,吸毒,寻衅滋事,建筑坍塌事故,强奸猥亵,意外伤亡,投放危险物质,抢劫抢夺,拐卖,挤压死伤事故,故意伤害,敲诈勒索,斗殴,杀人,求助,涉外事件,
    涉恐,涉枪,涉邪教,火灾,灾害事故,爆炸爆燃事故,疫情灾害,盗窃,纵火,组织卖淫,绑架,群众求助,自杀,自然天气灾害事故,诈骗,贩毒,赌博,走私,走私毒品,辟谣,阻
    碍执行职务,非法侵入住宅,非法制售爆炸物品,非法猎捕,非法盗伐林木,非法限制人身自由
    句子:2018年02月28日21时35分44秒报警人称:2018年2月28日21时30分许,报警人黄韦在本市鼓楼区吾悦广场门口接到订单,并在吾悦广场门口接到三位女性乘客,沿着
    湖北路往北走,行驶的速度很慢,快走到湖南路与马台街路口的位置时,大约有四个人把车子副驾驶一侧的反光镜拍弯后就往前走,报警人发现这几个喝多了就没理他门,
    继续往前走到红绿灯,刚过绿灯时,拍反光镜那几个人拦住报警人车子,其中一个戴金链子的男子趴在车前引擎盖上,并不断叫骂,另外三名男子,一个在拉副驾驶的车门
    、一个在给拍引擎盖的男子在拍照、还有一名男子站在旁边看。因为报警人车门没打开,三位醉酒男子便走开,于是报警人报警。
    上述句子中包含的事件类型包括:寻衅滋事
    '''
    pattern = "上述句子中包含的事件类型包括:(.{1,40})"

    structure = set()
    if special != text:
        pattern = re.compile(pattern)
        cls_list = pattern.findall(text)

        if len(cls_list):
            for cls_str in cls_list:
                cls_str = cls_str.replace("，",",")
                cls_str = cls_str.replace("。","")
                # cls_str = cls_str.replace("、",",")
                for cls in cls_str.split(","):
                    if cls in all_cls_list:
                        structure.add(cls)
                    else:
                        print("越界类别",cls)
    return structure


if __name__ == '__main__':
    # eval_file = "/data/wufan/data/NERData/relation/数据汇总/实验数据/人人关系/实验数据/chatglm_relation_p2p_fine_grit_v2_v0.6/valid_chatglm.json"
    # eval_file = "/data/wufan/experiments/llm/chatglm/relation_p2p/p2p_relation_v1.1.0/valid.json"
    # eval_file = "/data/wufan/experiments/llm/chatglm/relation_p2p/p2p_relation_v1.1.1/valid.json"
    eval_file = "/data/wufan/experiments/llm/chatglm/cls_jqtb/cls_jqtb_v1.0.0/valid.json"
    eval_file = "/data/wufan/experiments/llm/baichuan/cls_jqtb_v1.0.0/valid.json"
    eval_file = "/data/wufan/experiments/llm/chatglm/cls_jqtb/valid_chatgpt.json"

    eval_file = "/data/wufan/experiments/llm/chatglm2/cls_jqtb/cls_jqtb_v1.0.0/valid.json"
    all_num_tp, all_num_fp, all_num_fn = 0, 0, 0
    with jsonlines.open(eval_file) as reader:
        for line in reader:
            answer = line.get("answer","")
            target = line["target"]
            pairs_answer = text2structure(answer)
            pairs_target = text2structure(target)

            tp = pairs_target.intersection(pairs_answer)
            fp = pairs_answer.difference(pairs_target)
            fn = pairs_target.difference(pairs_answer)
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
