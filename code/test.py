import copy
import os
import openai
import json
import time
import random
from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import TrainingDatasetCRF
from sequence_aligner.containers import TraingingBatch
from transformers import BertTokenizerFast
#%%
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
# from util.process import ids_to_labels,Metrics,Metrics_e
from seqeval.scheme import BILOU

# def split_chinese_sentence(sentence):
#     return list(sentence)
#
# def add_space_between_chinese_characters(sentence):
#     result = ""
#     for char in sentence:
#         result += char + " "
#     return result.strip()
#
# sentence1 = "今天天气真好sss2.1"
# sentence2 = "GVK Power can go up to 14-15 levels : Vijay Bhambwani"
#
# print(list(sentence1))
# print(sentence1.split())
#
# print(list(sentence2))
# print(sentence2.split())
#
# print(add_space_between_chinese_characters(sentence1))
#
#
# def add_space_to_sentences(sentences_list):
#     new_list = []
#     for sentence in sentences_list:
#         new_sentence = ""
#         for char in sentence:
#             new_sentence += char + " "
#         new_list.append(new_sentence.strip())
#     return new_list
#
# sentences = ["今天天气真好", "我喜欢吃苹果"]
# print(add_space_to_sentences(sentences))




def list_to_str(list):
    list_copy = copy.deepcopy(list)  #深拷贝出一份新的列表，防止影响原列表
    for yuanzu in list_copy:
        # 去除'label'键
        yuanzu.pop('label', None)

        # 将字典转换为字符串并用'\n'连接
    result = '\n'.join(json.dumps(yuanzu, ensure_ascii=False) for yuanzu in list_copy)
    return result
list = [{'value': '长城汽车', 'start': 8, 'end': 12, 'tag': '正面', 'label': '正面'}, {'value': '长城汽车', 'start': 122, 'end': 126, 'tag': '正面', 'label': '正面'}]
print(list_to_str(list))

def contral_error_rate(data_path, error_index_list, compare_list, result_list):
    raw_data = json.load(open(data_path, encoding="utf-8"))

    # rate_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    rate_list = [1]
    # SEntFiN的训练集F1:【2383 0.43，  6018 0.53，  10830 0.59，  16983 0.65，  24342 0.69， 8602  0.81】
    for rate in rate_list:
        rate_data_list = []  # 存储真实标签
        for index2 ,item in enumerate(raw_data):
            print("00000000", item["output"], "\n", item["pre"])
            item["output"] = list_to_str(compare_list[index2]['annotations'])
            item["pre"] = list_to_str(result_list[index2]['annotations'])
            print("222222222", item["output"], "\n", item["pre"], "\n\n\n")
            if index2 in error_index_list:
                rate_data_list.append(item)
            else:
                if random.random() <= rate:
                    rate_data_list.append(item)
        # 将 compare_list 写入 JSON 文件
        train_pre_save_path = data_path.replace("save_pres", "train_pre22_"+str(rate))
        with open(train_pre_save_path, 'w', encoding='utf-8') as json_file:
            json.dump(rate_data_list, json_file, ensure_ascii=False, indent=4)

        print("正确样本保留",rate,"后的训练集预测数据，已成功写入", train_pre_save_path, "文件中")