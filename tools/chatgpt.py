import openai

openai.api_key = "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"

# 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

prompt = """明朝奸臣严嵩的儿子严世蕃每晨起身，痰唾很多，自蒙眬醒来至下床，唾壶须换去两三个。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "严嵩", "dst": "严世蕃", "relation": "儿子"}]



马曼琳，1992年1月9日出生于广东省潮阳，毕业于深圳广播电视大学，是腾讯创始人马化腾的女儿，任职于腾讯控股有限公司。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "马化腾", "dst": "马曼琳", "relation": "女儿"}]



后黄药师重收陆乘风归门，陆冠英亦得以随其父习桃花岛武功。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "黄药师", "dst": "陆乘风", "relation": "师生"}, {"src": "陆冠英", "dst": "陆乘风", "relation": "父母"}]



2019年9月26日6时30分左右，张俊伦因婚姻矛盾用锤子追打受害人韩艺，韩艺母亲路淑珍对其阻止

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "韩艺", "dst": "路淑珍", "relation": "母亲"}, {"src": "张俊伦", "dst": "韩艺", "relation": "配偶"}]



林一峰（林二汶的哥哥）、林嘉欣、黄耀明、at17的另一位成员卢凯彤以及林二汶的家人朋友分别给本书写了很特别的文字。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "林一峰", "dst": "林二汶", "relation": "兄弟姐妹"}]



凌雪雁，容貌秀丽，喜欢张君宝，与秦思容为情敌，后来由于得不到张君宝的爱慕，与爱她的宋远桥结为恋人。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[{"src": "凌雪雁", "dst": "宋远桥", "relation": "恋爱"}, {"src": "凌雪雁", "dst": "张君宝", "relation": "喜欢"}, {"src": "凌雪雁", "dst": "秦思容", "relation": "社会关系"}]



玛丽告诉彼得，她决定辞去工作去世界旅行。

# 抽取句子中所有人与人之间的关系，并以json格式输出

[]



这个星期末，我们打算去旅行，一起去的有吴、张、李和王四个人，我们都很期待这次旅行的经历。

# 抽取句子中所有人与人之间的关系，并以json格式输出

"""


prompt = "给一些警务方面的同义词"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
)
print(response["choices"][0]["message"]["content"])
