import json
import os
import numpy as np
import spacy
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch import nn
from tqdm import tqdm

device = torch.device("cuda:0")

# 文本处理
def get_text_dependency(text):
    # 加载Spacy模型
    nlp = spacy.load('en_core_web_sm')
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

    # chunk 没有用到
    # 构造chunk
    # chunks = [chunk.text for chunk in doc.noun_chunks]
    # result['chunk'] = chunks
    # result['chunk_cap'] = [chunk.text for chunk in doc.noun_chunks]
    #
    # # 构造chunk依赖
    # for i, chunk in enumerate(chunks):
    #     head_token = [token for token in doc if token.text == chunk]
    #     head_token = head_token[0]
    #     head_index = result['chunk_index'].index(doc[token.i].idx)
    #     result['chunk_dep'].append([head_index, i])
    #     result['chunk_index'].append(doc[doc[head_token.i].i].idx)

    return result

def split_image_into_blocks(img, block_size=32):
    width, height = img.size
    blocks = []
    for i in range(0, width, block_size):
        for j in range(0, height, block_size):
            block = img.crop((i, j, i + block_size, j + block_size))
            blocks.append(block)
    return blocks

# 读取数据集
data_path = 'dataset_our/gossipcop_v3-4_story_based_fake.json'
with open(data_path, "r", encoding='utf8') as f:
    dataset = json.load(f)

# 设置你的文件夹路径
folder_path = 'dataset_our/top_img/'
entries = os.listdir(folder_path)

# 判断所有数据是否都包含图片
#共968条数据没有图片，删除这些数据s
dataset_filter = {}  # 共2783条数据
for idx, key in tqdm(enumerate(dataset)):
    data = dataset[key]
    # flag = data['has_top_img']
    # if flag == 0:
    #     count += 1
    # 检查是否存在名为'a'的图片文件
    img_name = data['origin_id']
    for entry in entries:
        # 检查文件名是否以'a'开头，并且是图片格式
        # if entry.startswith(img_name) and os.path.splitext(entry)[1] in ['.jpg', '.png', '.jpeg', '.gif', '.bmp']:
        if entry.startswith(img_name) and os.path.splitext(entry)[1] in ['.png']:
            img_path = folder_path + img_name + '_top_img.png'
            try:
                with Image.open(img_path) as img:
                    img.verify()  # 这将验证文件的合法性
                dataset_filter[key] = data
            except (IOError, SyntaxError) as e:
                print(f"Error opening the image: {e}")
                print(img_path)

'''
dataset format:
twitter:
[twitter_id, text, img_id('sandyA_fake_29.jpg'), label, caption, dependency of text, dependency of caption]
 {'chunk_cap': ['scary shit', '#', 'hurricane', '#', 'ny'],
  'token_cap': ['scary', 'shit', '#', 'hurricane', '#', 'ny'],
  'token_dep': [[0, 1], [3, 1], [5, 3]],
  'chunk_dep': [[2, 0], [4, 2]],
  'chunk': ['scary shit', 'hurricane', 'ny'],
  'chunk_index': [0, 2, 4]},
'''

new_dataset = []
pad = None  # 作为无用内容的填充
for idx, key in tqdm(enumerate(dataset_filter)):
    data = dataset_filter[key]
    temp_data = []
    temp_data.append(data['origin_id'])  # id
    temp_data.append(data['origin_title'])  # text
    temp_data.append(data['origin_id'])  # img_id
    if data['origin_label'] == 'legitimate':  # label
        temp_data.append(1)
    else:
        temp_data.append(0)
    temp_data.append(pad)  # caption：没有用到

    # 文本处理
    dependency_of_text = get_text_dependency(data['origin_title'])
    temp_data.append(dependency_of_text)  # # dependency of text

    temp_data.append(pad)  # dependency of caption： 没有用到
    new_dataset.append(temp_data)

# 使用UTF-8编码打开文件以写入
with open('dataset_our/gossipcop_v3-4_story_based_fake_pre-process.json', 'w', encoding='utf-8') as json_file:
    # 使用json.dump将字典写入文件，并设置缩进以美化输出
    json.dump(new_dataset, json_file, ensure_ascii=False, indent=4)
print('new dataset had been saved.')

# 图片的编码
# 加载预训练的ResNet-34模型
# resnet34 = models.resnet34(pretrained=True)
# resnet34.fc = nn.Linear(512, 512)
# resnet34.eval()
# # resnet34 = resnet34.to(device)
#
# img_dic = {}
# for idx, key in tqdm(enumerate(dataset_filter)):
#     data = dataset_filter[key]
#     img_name = data['origin_id']
#     img_path = folder_path + img_name + '_top_img.png'
#     img = Image.open(img_path).convert('RGB')
#     img = img.resize((224, 224))
#     assert img.mode == 'RGB', 'not RGB'
#     blocks = split_image_into_blocks(img)
#
#     # 定义转换操作，转换为Tensor并进行归一化
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     # 存储特征向量的列表
#     features = []
#
#     # 对每个块进行编码
#     for block in blocks:
#         # 将PIL图像转换为Tensor
#         block_tensor = transform(block).unsqueeze(0)  # 添加批次维度
#         with torch.no_grad():
#             # 获取特征向量
#             block_feature = resnet34(block_tensor)
#         features.append(block_feature.squeeze(0).cpu().numpy())  # 移除批次维度并转换为NumPy数组
#     # 将49个特征向量合并为一个数组
#     all_features = np.array(features)
#     img_embedding = torch.tensor(all_features)
#     img_dic[img_name] = img_embedding
#
# torch.save(img_dic, 'dataset_our/embedding34_our.pt')
# print('embedding34_our.pt had been saved.')
print('ok')
