
import jsonlines
import json
path = "/data/wufan/data/wx_bilu_aug/val_aug_0609.json"
result = []
c = 0
instruction = """###Instruction:

根据句子内容，针对句子中未提问的问题或者已经提到的事情进一步提问，返回几个的提问的结果，并满足如下几点要求：
1.如果在提问并回答如下话题：个人情况，个人简历，家庭成员，法律条款，身体状况等，返回的问题可以参考但不局限于：因为什么事情报案？描述一下具体事情发生的经过？
2.如果在提问并回答案件经过，需要依据人物，时间，地点，事件内容，补充句子中未提及的问题；
3.如果事发经过中，未提及事情发生的时间、地点，请补充提问；
4.不能提问与句子无关的内容；
5.不需要回答句子中的问题；
6.问题在对话中不能有答案；
7.问题需要对警察梳理案件有正向促进作用；
8.不能提问句子中已经存在或者相似的问题；
9.提问5~10个问题，每个问题不少于15字\n"""

with jsonlines.open(path) as reader:
    for line in reader:
        input = line["input"]
        x = \
            f"""###Instruction:

根据句子内容，针对句子中未提问的问题或者已经提到的事情进一步提问，返回几个的提问的结果，并满足如下几点要求：
1.如果在提问并回答如下话题：个人情况，个人简历，家庭成员，法律条款，身体状况等，返回的问题可以参考但不局限于：因为什么事情报案？描述一下具体事情发生的经过？
2.如果在提问并回答案件经过，需要依据人物，时间，地点，事件内容，补充句子中未提及的问题；
3.如果事发经过中，未提及事情发生的时间、地点，请补充提问；
4.不能提问与句子无关的内容；
5.不需要回答句子中的问题；
6.问题在对话中不能有答案；
7.问题需要对警察梳理案件有正向促进作用；
8.不能提问句子中已经存在或者相似的问题；
9.提问5~10个问题，每个问题不少于15字

###Input:

{input}"""
        x = x.replace("可以继续提问的1个问题：\n","").replace("可以继续提问的2个问题：\n","").replace("可以继续提问的3个问题：\n","")
        # print(x)
        input = input.replace("可以继续提问的1个问题：","").replace("可以继续提问的2个问题：","").replace("可以继续提问的3个问题：","")
        result.append({"instruction": instruction, "input": input})
        print(input)
        print()
        c+=1
        if c==100:
            break

#
#
# output_path = "../wx1.json"
# with open(output_path,'w',encoding="utf-8") as w:
#     json.dump(result[:1000],w,ensure_ascii=False)
#
# output_path = "../wx2.json"
# with open(output_path,'w',encoding="utf-8") as w:
#     json.dump(result[1000:],w,ensure_ascii=False)