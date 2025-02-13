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

def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list

    # Correcting start and end tags    对 result_list 中的每个元素进行处理，修正起始和结束标签。
    #代码确保了注释在文本中的位置不重叠，并且更新了注释的起始位置和结束位置
def re_position(result_list):
    for i, item in enumerate(result_list):
        text = item['content']
        annos = item['annotations']  #{}{}
        sorted_annos = sorted(annos, key=lambda x: x['start']) #按start值对每组排序
        value_list = []
        start_list = []
        last_start = -1
        drop_list = []
        for indx, sub_annos in enumerate(sorted_annos): #对于每组{"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}

            value = sub_annos['value']   #"value": "Reddit Inc"
            # if value not in value_list:
            #     start = text.find(value)
            # else:
            #     index_list = []
            #     for j, v in enumerate(value_list):
            #         if v == value:
            #             index_list.append(j)
            #     sub_start = subset(start_list, index_list)
            #     last_start = max(sub_start)  #使用 max(sub_start) 来找到子列表中的最大值，即之前相同值的最后一个起始位置，然后将其加1作为新的起始位置。
            #     start = text.find(value, last_start + 1)  #保证新的注释不会与之前的注释重叠
            start = text.find(value, last_start+1)
            sub_annos['start'] = start
            sub_annos['end'] = start + len(value)
            last_start = start + len(value)

            value_list.append(value)
            start_list.append(start)
    return result_list


def get_compare(data_path):
    raw = json.load(open(data_path, encoding="utf-8"))
    compare_list = []  #存储真实标签
    for item in raw:
        comp = {}
        comp["content"] = item["content"]
        comp_anno_list = []
        comp_strs = item["output"].split('\n')
        for res in comp_strs:  # 对于每一个实体组 {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}
            index_left = res.find('{')
            index_right = res.find('}')
            if index_right == -1 or index_left == -1:
                continue
            res = res[index_left:index_right + 1]
            sub_json = json.loads(res)
            comp_anno_list.append(sub_json)
        comp["annotations"] = comp_anno_list
        compare_list.append(comp)
        compare_list = re_position(compare_list)  #有的句子中含有非法字符，需要重新确定一下位置，以与预测标签可以对应。

    # pre_raw = json.load(open(data_path+'save_pres_FinEntity_2024-04-03-21-02-48.json'))
    pre_raw = json.load(open(data_path, encoding="utf-8"))
    result_list= []  #存储预测标签
    for it in pre_raw:
        result = {}
        result["content"] = it["content"]
        anno_list = []
        pre_strs = it["pre"].split('\n')
        # pre_strs = it["output"].split('\n')
        for res in pre_strs:  #对于每一个实体组 {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}
            index_left = res.find('{')
            index_right = res.find('}')
            if index_right == -1 or index_left == -1:
                continue
            res = res[index_left:index_right + 1]
            sub_json = json.loads(res)
            anno_list.append(sub_json)
        result["annotations"] = anno_list

        result_list.append(result)

    print("LLM预测结果中annotations：", result_list[11]["annotations"])
    result_list = re_position(result_list)
    print("修正起始位置后中annotations：", result_list[11]["annotations"])
    print("测试集中真实的 annotations：", compare_list[11]["annotations"])

    print(len(compare_list))
    print(len(result_list))
    return compare_list, result_list

def check_anno(compare_list, result_list):
    # %%找到 result_list 中所有具有负起始位置的注释，然后打印包含这些注释的整个字典项，以便进行进一步的检查或处理
    for idx, item in enumerate(result_list):
        annos = item['annotations']
        for a in annos:
            if a['start'] < 0:
                print("result_list:", item)

    # %%只保留起始位置大于等于 0 的组
    for idx, item in enumerate(result_list):
        annos = item['annotations']
        drop_list = []
        item['annotations'] = [x for x in annos if x['start'] >= 0]

    for example in result_list:
        for annotation in example['annotations']:
            # We expect the key of label to be label but the data has tag
            annotation['label'] = annotation['tag']

    # %%找到 result_list 中所有具有负起始位置的注释，然后打印包含这些注释的整个字典项，以便进行进一步的检查或处理
    for idx, item in enumerate(compare_list):
        annos = item['annotations']
        for a in annos:
            if a['start'] < 0:
                print("compare_list:", item)

    # %%只保留起始位置大于等于 0 的组 , 原数据集中有一条数据的二元组有重复导致reposition后x['start']=-1，导致报错，compare_list经过此函数就可以不报错。
    for idx, item in enumerate(compare_list):
        annos = item['annotations']
        drop_list = []
        item['annotations'] = [x for x in annos if x['start'] >= 0]

    for example in compare_list:
        for annotation in example['annotations']:
            # We expect the key of label to be label but the data has tag
            annotation['label'] = annotation['tag']

    return compare_list, result_list

def error_analyze(data_path, compare_list, result_list):
    print("\n\n\n#################################错误样本分析#####################################\n\n\n")
    # print(compare_list[0])
    # print(result_list[0])
    # 打开一个txt文件用于写入
    error_index_list = []
    error_path = data_path.replace("save_pres", "error_analyze_save_pres").replace(".json", ".txt")
    with open(error_path, 'w', encoding='UTF-8') as file:
        print(compare_list[0])

        for index, (item1, item2) in enumerate(zip(compare_list, result_list)):

            item1['annotations'] = sorted(item1['annotations'], key=lambda x: x['start'])

            if item1['annotations'] != item2['annotations']:
                error_index_list.append(index)
                # print(str(item1['content']))
                file.write(str(item1['content']) + "\n")
                # print("真实标签:", item1['annotations'])
                file.write("真实标签:" + str(item1['annotations']) + "\n")
                # print("预测标签:", item2['annotations'])
                file.write("预测标签:" + str(item2['annotations']) + "\n")
                # print()
                file.write("\n")

    print("错误样本已写入", error_path, "文件中")
    print("error_index_list长度:", len(error_index_list))
    return error_index_list

#[{'value': "Macy's", 'start': 32, 'end': 38, 'tag': 'Positive', 'label': 'Positive'}, {'value': 'Walmart', 'start': 188, 'end': 195, 'tag': 'Negative', 'label': 'Negative'}]

def list_to_str(list):
    list_copy = copy.deepcopy(list)  #深拷贝出一份新的列表，防止影响原列表
    for yuanzu in list_copy:
        # 去除'label'键
        yuanzu.pop('label', None)

        # 将字典转换为字符串并用'\n'连接
    result = '\n'.join(json.dumps(yuanzu, ensure_ascii=False) for yuanzu in list_copy)
    return result

def contral_error_rate(data_path, error_index_list, compare_list, result_list):
    raw_data = json.load(open(data_path, encoding="utf-8"))

    # rate_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    rate_list = [1]
    # SEntFiN的训练集F1:【2383 0.43，  6018 0.53，  10830 0.59，  16983 0.65，  24342 0.69， 8602  0.81】
    for rate in rate_list:
        rate_data_list = []  # 存储真实标签
        for index2 ,item in enumerate(raw_data):
            # print("00000000", compare_list[index2]['annotations'], "\n")
            item["output"] = list_to_str(compare_list[index2]['annotations'])
            item["pre"] = list_to_str(result_list[index2]['annotations'])
            # print("222222222", item["output"], "\n")
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


def report(data_path, compare_list, result_list):
    # 计算得分
    # tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-pretrain')
    tokenizer = BertTokenizerFast.from_pretrained('finbert')
    if('FinEntity' in data_path):
        label_set = LabelSet(labels=["Neutral", "Positive", "Negative"])  # label in FinEntity
    elif('SEntFiN' in data_path):
        label_set = LabelSet(labels=["neutral", "positive", "negative"])  # label in SEntFiN
    elif ('EFSA' in data_path):
        label_set = LabelSet(labels=["中立", "正面", "负面"])# label in EFSA

    dataset = TrainingDatasetCRF(data=compare_list, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=128)
    dataset_openai = TrainingDatasetCRF(data=result_list, tokenizer=tokenizer, label_set=label_set,
                                        tokens_per_batch=128)

    label_list = []
    pred_label_list = []
    for i in range(len(dataset)):
        sub_list = []
        pred_sub_list = []

        for m in dataset[i].labels:
            if m == -1:
                continue
            else:
                sub_list.append(label_set.ids_to_label[m])

        if i == 0:
            print(dataset[i])
            print(dataset[i].labels)
            print(sub_list)

        for n in dataset_openai[i].labels:
            if n == -1:
                continue
            else:
                if n == None:
                    n = 0
                pred_sub_list.append(label_set.ids_to_label[n])
        label_list.append(sub_list)
        pred_label_list.append(pred_sub_list)

    report = classification_report(label_list, pred_label_list, mode='strict', scheme=BILOU)
    print(report)

# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-22-23-30-noicl.json"  #NOICL
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-22-18-06-noicl.json"  #NOICL
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-02-21-58-48.json"  #rd1
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-20-24-03.json" #rd2
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-20-29-28.json" #rd2
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-23-30-20.json" #rd2
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-23-35-46.json/" #rd2
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-21-17-40.json"  #rd3
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-06-21-23-17.json"  #rd3
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-07-00-59-04.json"  #rd3
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-07-02-36-24.json"  #rd4
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-07-04-43-20.json"  #rd5
# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-07-04-57-46.json" #rd5
# data_path = "../llm_result/result1/"

# data_path = "../llm_data/result1/2save_pres_FinEntity_2024-04-07-20-17-56.json"   #fin+fix+3e
# data_path = "../llm_data/result1/2save_pres_FinEntity_2024-04-07-20-23-29.json"   #fin+fix+3e
# data_path = "../llm_data/result1/2save_pres_FinEntity_2024-04-07-20-58-05.json"   #fin+rd+3e
# data_path = "../llm_data/result1/2save_pres_FinEntity_2024-04-07-21-03-44.json"   #fin+rd+3e
# data_path = "../llm_result/result1/sort_save_pres_FinEntity_2024-04-08-23-28-52.json"   #fin+fix+3e +sort
# data_path = "../llm_result/result1/sort_save_pres_FinEntity_2024-04-08-23-34-40.json"   #fin+fix+3e +sort
# data_path = "../llm_data/result1/sort_save_pres_FinEntity_2024-04-09-00-08-41.json"  #fin+rd+3e +sort
# data_path = "../llm_data/result1/sort_save_pres_FinEntity_2024-04-09-00-14-37.json"  #fin+rd+3e +sort
# data_path = "../llm_data/result1/save_pres_FinEntity_2024-04-09-09-16-19.json"    #fin+fix+3e +sort复现
# data_path = "../llm_data/result1/save_pres_FinEntity_2024-04-09-09-22-02.json"    #fin+fix+3e +sort复现
# data_path = "../llm_data/result1/save_pres_FinEntity_2024-04-09-09-55-03.json"    #fin+rd+3e +sort 修正prompt
# data_path = "../llm_data/result1/save_pres_FinEntity_2024-04-09-10-00-49.json"    #fin+rd+3e +sort 修正prompt

# data_path = "../llm_data/result2/save_pres_FinEntity_2024-04-09-11-42-30.json"#fin+fix+3e +sort

# data_path = "../llm_data/result2/save_pres_FinEntity_2024-04-09-11-48-08.json"# #fin+fix+3e +sort

# data_path = "../llm_data/result2/save_pres_FinEntity_2024-04-09-12-22-28.json"# # #fin+rd+3e +sort 修正prompt

# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-09-09-33-16.json"# # # #fin+rd+3e +sort 修正prompt


# data_path = "../llm_result/result1/save_pres_FinEntity_2024-04-09-10-00-49.json"


# data_path = "../llm_result/SEntFin/save_pres_SEntFiN_2024-05-29-18-26-47.json"
# data_path = "../llm_result/EFSA/save_pres_EFSA_2024-05-30-04-49-12.json"

# data_path = "../IstructABSA_result/finentity/save_pres_FinEntity_result_eval.json"
# data_path = "../IstructABSA_result/sentfin/save_pres_SEntFiN_result_eval.json"

# data_path = "../train_pre/SEntFiN/save_pres_SEntFiN_2024-07-08-17-30-08.json"       #8602  0.81
# data_path = "../train_pre/SEntFiN/save_pres_SEntFiN_test_org.json"       #

# data_path = "../train_pre/SEntFiN/train_pre_0_SEntFiN_2024-07-08-17-30-08.json"     #2383 0.43
# data_path = "../train_pre/SEntFiN/train_pre_0.2_SEntFiN_2024-07-08-17-30-08.json"   #3602 0.59
# data_path = "../train_pre/SEntFiN/train_pre_0.4_SEntFiN_2024-07-08-17-30-08.json"   #4858 0.69
# data_path = "../train_pre/SEntFiN/train_pre_0.6_SEntFiN_2024-07-08-17-30-08.json"   #6169  0.75
# data_path = "../train_pre/SEntFiN/train_pre_0.8_SEntFiN_2024-07-08-17-30-08.json"     #7373 0.78

# data_path = "../train_pre/EFSA/save_pres_EFSA_2024-07-08-18-58-04.json"   #8665  0.92
# data_path = "../train_pre/EFSA/save_pres_EFSA_test_org.json"   #

# data_path = "../train_pre/EFSA/train_pre_0.9_EFSA_2024-07-08-18-58-04.json"
# data_path = "../train_pre/EFSA/train_pre_0.7_EFSA_2024-07-08-18-58-04.json"
# data_path = "../train_pre/EFSA/train_pre_0.5_EFSA_2024-07-08-18-58-04.json"
# data_path = "../train_pre/EFSA/train_pre_0_EFSA_2024-07-08-18-58-04.json"      #621 0.12
# data_path = "../train_pre/EFSA/train_pre_0.2_EFSA_2024-07-08-18-58-04.json"    #2231 0.71
# data_path = "../train_pre/EFSA/train_pre_0.4_EFSA_2024-07-08-18-58-04.json"    #3901 0.83
# data_path = "../train_pre/EFSA/train_pre_0.6_EFSA_2024-07-08-18-58-04.json"    #5415 0.88
# data_path = "../train_pre/EFSA/train_pre_0.8_EFSA_2024-07-08-18-58-04.json"    #7058 0.91

# data_path = "../correct_result/EFSA/save_pres_EFSA_correct_2024-08-30-21-51-36.json"
data_path = "../correct_result/EFSA/save_pres_EFSA_correct_2024-09-02-09-28-45.json"



# data_path = "../train_pre/FinEntity/save_pres_FinEntity_2024-07-10-10-36-28.json"       #783 0.89
# data_path = "../train_pre/FinEntity/save_pres_FinEntity_test_org.json"
# data_path = "../train_pre/FinEntity/train_pre_0_FinEntity_2024-07-10-10-36-28.json"     #148 0.53
# data_path = "../train_pre/FinEntity/train_pre_0.2_FinEntity_2024-07-10-10-36-28.json"   #274 0.71
# data_path = "../train_pre/FinEntity/train_pre_0.4_FinEntity_2024-07-10-10-36-28.json"   #411 0.80
# data_path = "../train_pre/FinEntity/train_pre_0.6_FinEntity_2024-07-10-10-36-28.json"   #532 0.85
# data_path = "../train_pre/FinEntity/train_pre_0.8_FinEntity_2024-07-10-10-36-28.json"   #647 0.87


# data_path = "../correct_result/SEntFiN/save_pres_SEntFiN_correct_2024-07-10-03-03-26.json"
# data_path = "../correct_result/FinEntity/save_pres_FinEntity_correct_2024-07-10-12-00-17.json"  #fix
# data_path = "../correct_result/EFSA/save_pres_EFSA_correct_2024-07-12-04-38-59.json"


# data_path = "../correct_result/FinEntity/save_pres_FinEntity_correct_2024-07-23-20-12-21.json"   #gnn
# data_path = "../correct_result/SEntFiN/save_pres_SEntFiN_correct_2024-07-24-05-01-17.json"


#加上正误信息

# data_path = "../correct_result/FinEntity/save_pres_FinEntity_correct_2024-07-31-19-06-21.json"  #1
# data_path = "../correct_result/FinEntity/save_pres_FinEntity_correct_2024-07-31-20-43-14.json"  #0.8








compare_list, result_list = get_compare(data_path)  # 获取真实标签和预测标签并修正位置
compare_list, result_list = check_anno(compare_list, result_list)  #检查不合规的位置信息，只保留起始位置大于等于 0 的组

# error_index_list = error_analyze(data_path, compare_list, result_list)  #错误分析,将错误的样本写入txt文件
# contral_error_rate(data_path, error_index_list, compare_list, result_list)# 随机删除部分正确样本，使训练集错误率和测试集相当

report(data_path, compare_list, result_list)  #报告序列标注得分

