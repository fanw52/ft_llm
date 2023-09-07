#utf-8
import openai
import time
from tqdm import tqdm
import json
import jsonlines
# openai.api_key = "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"
openai.api_key = "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"


result = []

prompt = """###Instruction:

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

问:这是《证人权利义务告知书》,交给你阅读,如果你阅读有困难,我们可以向你宣读。如果你有不明白的地方,我们可以向你解释。
答:好的。
问:你的身份情况?
答:我叫凌辰轩,重庆市鞍山市华容县新河镇山泉村山泉路00号山泉新村000号族,高中,居民身份证号码:110110194107032222,出生日期:2002年07月01日,户籍地址:重庆市鞍山市华容县新河镇山泉村山泉路00号山泉新村000号,现住址:重庆市鞍山市华容县新河镇山泉村山泉路00号山泉新村000号,工作单位是:苏州大学联系电话:13700000000。
问:你因何事来公安机关?
答:我们因车辆行驶,与人起了纠纷。
问:你把事情经过讲一下?
答:2023年01月28日晚上8点左右,我驾驶着车辆在澄杨路山泉路口上,从东往西行驶,路口是三车道,我们在左转车道上面(车上当时5个人,我,我爸妈,堂叔和堂叔妈妈)。驾驶到前方红绿灯路口,我因离合没踩稳,导致车子熄火,这时前方信号灯已经转绿,但是我打了三次火都没成功,后面的车辆已经在鸣喇叭示意了,我老爸探出窗向后面示意,说了句抱款,打了声招呼,到信号灯快转红我才把车开走,对方跟着我们过了路口,我们把车子从西面大门进了山泉新村小区,去停到大西桥山泉村山泉路00号山泉新村000栋东面的停车位上,这时一个男人从后面一栋楼跟过来看上去是我们在红绿灯路口时,后面一辆车上的人),问我们当时在路口为什么停着不动,还用力拍打我的车玻璃,我家里人就都下车,与对方解释缘由,但对方态度比较差,我老爸的脾气也比较硬
"""
c = 0
with jsonlines.open("./wx1_chatgpt.jsonl",'w') as w:
    with open("./wx1.json",encoding="utf-8") as rfile:
        data = json.load(rfile)
        for line in tqdm(data):
            try:
                prompt = line["instruction"] + line["input"]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                print(response["choices"][0]["message"]["content"])
                result.append({"answer": response["choices"][0]["message"]["content"]})
                line["target"] = response["choices"][0]["message"]["content"]
                w.write(line)
            except Exception as e:
                print(e)
                pass
            time.sleep(20)

# with open("../data/data.json", 'w', encoding="utf-8") as w:
#     json.dump(result,w,ensure_ascii=False,indent=2)
