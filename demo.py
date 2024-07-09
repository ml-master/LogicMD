import json

import spacy

# 加载Spacy模型
nlp = spacy.load('en_core_web_sm')

# 读取数据集
data_path = 'dataset_our/gossipcop_v3-4_story_based_fake.json'
with open(data_path, "r", encoding='utf8') as f:
    dataset = json.load(f)

# 判断所有数据是否都包含图片
#共968条数据没有图片，删除这些数据
count = 0
for idx, key in enumerate(dataset):
    data = dataset[key]
    flag = data['has_top_img']
    if flag == 0:
        count += 1
print(count)
# dataset_filter = {k: v for k, v in dataset.items() if v['has_top_img'] != 0}

# 保存筛选后的数据
# with open('dataset_our/gossipcop_v3-3_integration_based_fake_tn200_pre-process.json', 'w', encoding='utf8') as json_file:
#     json.dump(new_dict, json_file, ensure_ascii=False, indent=4)

# 构造数据集
for idx, key in enumerate(dataset_filter):
    data = dataset_filter[key]
    temp_dict['news_id'] = data[]


# 待处理的文本
text = "scary shit #hurricane #ny"

# 使用模型处理文本
doc = nlp(text)

# 初始化结果字典
result = {
    'chunk_cap': [],
    'token_cap': [],
    'token_dep': [],
    'chunk_dep': [],
    'chunk': [],
    'chunk_index': []
}

# 遍历文本中的token
for token in doc:
    result['token_cap'].append(token.text)

# 构造token依赖列表
for token in doc:
    if token.dep_ != 'ROOT':  # 忽略句子的根节点
        head_index = result['token_cap'].index(token.head.text)
        result['token_dep'].append([head_index, result['token_cap'].index(token.text)])

# 构造chunk
chunks = [chunk.text for chunk in doc.noun_chunks]
result['chunk'] = chunks
result['chunk_cap'] = [chunk.text for chunk in doc.noun_chunks]

# 构造chunk依赖
for i, chunk in enumerate(chunks):
    head_token = [token for token in doc if token.text == chunk]
    head_token = head_token[0]
    head_index = result['chunk_index'].index(doc[token.i].idx)
    result['chunk_dep'].append([head_index, i])
    result['chunk_index'].append(doc[doc[head_token.i].i].idx)

print(result)
