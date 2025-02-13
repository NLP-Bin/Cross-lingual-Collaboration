import os
import openai
import json
import time


# %%
def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list





def gpt(num):
    raw = json.load(open('../data/input/EFSA_test.json', encoding='utf8'))
    raw = raw[:num]
    openai.api_key = "sk-oGppXnowHQwOBrwL93945dDe99124523B240Da28061e78Ad"
    openai.api_base = "https://www.jcapikey.com/v1"

    sys_prompt = "摒弃之前所有的指令。你现在是一个中文文本金融实体识别器和情感分类器。 "
    user_prompt = """识别给定句子中属于公司或组织的金融实体，并根据句意给出每个实体段的情感极性，情感分类为“中立”、“正面”或“负面”。
        考虑把每句话都当作一个字符串，使用从零开始的索引为实体提供起始和结束索引来标记其边界，包括空格和标点符号。
    在输出中，value表示实体名称，请提供实体的起始和结束索引以标记其边界，Tag表示情感类别。
    输出四元组格式示例：{\"value\": \"上港集团\", \"start\": 0, \"end\": 4, \"tag\": \"正面\"}，用换行符间隔不同的四元组。
    句子中可能包含不同个数的金融实体，请注意输出格式，不要输出“客户”，“公司”等笼统的无关内容。
            """
#     user_prompt = """识别以下内容中属于公司或组织的金融实体，并给出句子中出现的每个实体段的情感极性，情感分类为“中立”、“正面”或“负面”。
#     考虑把每句话都当作一个字符串，使用从零开始的索引为实体提供起始和结束索引来标记其边界，包括空格和标点符号。
# 在输出中，value表示实体名称，请提供实体的起始和结束索引以标记其边界，Tag表示情感类别。
# 句子："中银证券04月15日发布研报称，维持恩捷股份(002812.SZ，最新价：183.78元)增持评级。评级理由主要包括：1)预计2022Q1业绩增长超100%；2)成本优势不断强化，经营现金流持续向好；3)产能布局领先，产品结构持续优化。"
#         """
    # 输出格式示例：{\"value\": \"上港集团\", \"start\": 0, \"end\": 4, \"tag\": \"正面\"}\n{\"value\": \"上港集团\", \"start\": 25, \"end\": 29, \"tag\": \"正面\"}
    # 句子中可能包含不同个数的实体，请注意输出格式。
    assist_prompt = """{\"value\": \"恩捷股份\", \"start\": 18, \"end\": 22, \"tag\": \"正面\"}
        """
    user_prompt2 = """长城微光(08286.HK)公告,韩蕾为投入更多时间于其他业务及个人事务,已辞任公司公司秘书及授权代表,自2021年5月6日起生效。
        """
    assist_prompt2 = """{\"value\": \"长城微光\", \"start\": 0, \"end\": 4, \"tag\": \"中立\"}
        """

    user_prompt3 = """上港集团01月04日陆股通净买入561.42万元。上港集团近5日陆股通资金呈现持续买入状态，近5日北上资金累计净买入8558.69万元，外资近期有持续流出的迹象。
        """

    assist_prompt3 = """{\"value\": \"上港集团\", \"start\": 0, \"end\": 4, \"tag\": \"正面\"}\n{\"value\": \"上港集团\", \"start\": 25, \"end\": 29, \"tag\": \"正面\"}
        """

    result_list = []
    compare_list = []
    for item in raw:
        sentence = item['content']
        print(sentence)
        # sentence ='Nearly all major S&P 500 sectors are red, with materials <.SPLRCM> and communications services <.SPLRCL> taking the biggest hits. Staples <.SPLRCS> and healthcare <.SPXHC> are posting small gains.'
        try:
            rsp = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                    # {"role": "assistant", "content": assist_prompt},
                    # {"role": "user", "content": user_prompt2},
                    # {"role": "assistant", "content": assist_prompt2},
                    # {"role": "user", "content": user_prompt3},
                    # {"role": "assistant", "content": assist_prompt3},
                    {"role": "user", "content": sentence},
                ],
                temperature=0.0,
            )
        except:  #未获取到结果，再次获取
            print("retry request")
            time.sleep(2)
            rsp = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4o",
                messages=[{"role": "user", "content": sentence}, ], temperature=0.0, )
        else:   #成功获取到结果
            result_dict = {}
            result_dict['content'] = sentence

            # 如果获取结果为空
            if len(rsp['choices']) == 0:
                result_dict['annotations'] = []
                result_list.append(result_dict)
                compare_list.append(item)
                continue
            # 如果获取结果不为空
            choice = rsp['choices'][0]
            message = choice['message']
            res_str = message['content']


            ########从这里



            res_str = res_str.split('\n')
            #以\n区分不同的实体组  {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}\n{"start": 128, "end": 138, "value": "Intel Corp", "tag": "Neutral"}
            anno_list = []
            #如果message['content']为空
            if len(res_str) == 0:
                result_dict['annotations'] = []
                result_list.append(result_dict)
                compare_list.append(item)
                continue
            for res in res_str:  #对于每一个实体组 {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}
                index_left = res.find('{')
                index_right = res.find('}')
                if index_right == -1 or index_left == -1:
                    continue
                res = res[index_left:index_right + 1]
                sub_json = json.loads(res)
                anno_list.append(sub_json)
            result_dict['annotations'] = anno_list
            result_list.append(result_dict)
            compare_list.append(item)
    with open('../data/output/4o-open_ai_EFSA_0_911.json', 'wt', encoding='utf8') as f:
        print(json.dumps(result_list, indent=2, ensure_ascii=False), file=f)

    # Correcting start and end tags    对 result_list 中的每个元素进行处理，修正起始和结束标签。
    #代码确保了注释在文本中的位置不重叠，并且更新了注释的起始位置和结束位置
    for i, item in enumerate(result_list):
        text = item['content']
        annos = item['annotations']  #{}{}
        if (annos == [{}]):
            item['annotations'] = [{
                "start": 0,
                "end": 0,
                "value": "aaaaa",
                "tag": "正面"
            }]
            sorted_annos = item['annotations']
        else:
            sorted_annos = sorted(annos, key=lambda x: x['start']) #按start值对每组排序
        value_list = []
        start_list = []
        drop_list = []
        for indx, sub_annos in enumerate(sorted_annos): #对于每组{"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}

            value = sub_annos['value']   #"value": "Reddit Inc"
            if value not in value_list:
                start = text.find(value)
            else:
                index_list = []
                for j, v in enumerate(value_list):
                    if v == value:
                        index_list.append(j)
                sub_start = subset(start_list, index_list)
                last_start = max(sub_start)  #使用 max(sub_start) 来找到子列表中的最大值，即之前相同值的最后一个起始位置，然后将其加1作为新的起始位置。
                start = text.find(value, last_start + 1)  #保证新的注释不会与之前的注释重叠
            sub_annos['start'] = start
            sub_annos['end'] = start + len(value)
            value_list.append(value)
            start_list.append(start)

    print("COMPLETE")

    #%%
    print(len(compare_list))
    print(len(result_list))
    return compare_list, result_list


def get_compare(num):
    raw = json.load(open('../data/input/EFSA_test.json', encoding='utf8'))
    raw = raw[:num]
    compare_list = []
    for item in raw:
        compare_list.append(item)
    result_list = json.load(open('../data/output/4o-open_ai_EFSA_0_911.json', encoding='utf8'))
    print(len(compare_list))
    print(len(result_list))

    for i, item in enumerate(result_list):
        text = item['content']
        annos = item['annotations']  # {}{}
        print(annos)
        if (annos == [{}]):
            item['annotations'] = [{
                "start": 0,
                "end": 0,
                "value": "aaaaa",
                "tag": "Positive"
            }]
            sorted_annos = item['annotations']
        else:
            sorted_annos = sorted(annos, key=lambda x: x['start'])  # 按start值对每组排序
        value_list = []
        start_list = []
        drop_list = []
        for indx, sub_annos in enumerate(
                sorted_annos):  # 对于每组{"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}

            value = sub_annos['value']  # "value": "Reddit Inc"
            if value not in value_list:
                start = text.find(value)
            else:
                index_list = []
                for j, v in enumerate(value_list):
                    if v == value:
                        index_list.append(j)
                sub_start = subset(start_list, index_list)
                last_start = max(sub_start)  # 使用 max(sub_start) 来找到子列表中的最大值，即之前相同值的最后一个起始位置，然后将其加1作为新的起始位置。
                start = text.find(value, last_start + 1)  # 保证新的注释不会与之前的注释重叠
            sub_annos['start'] = start
            sub_annos['end'] = start + len(value)
            value_list.append(value)
            start_list.append(start)
    with open('../data/output/4o-open_ai_EFSA_0_911_result.json', 'wt', encoding='utf8') as f:
        print(json.dumps(result_list, indent=2, ensure_ascii=False), file=f)
    print("COMPLETE")

    # %%
    print(len(compare_list))
    print(len(result_list))
    return compare_list, result_list


num = 200
# compare_list, result_list = gpt(num)

# compare_list, result_list = get_compare(num)
compare_list, result_list = gpt(num)



#%%只保留起始位置大于 0 的组
for idx,item in enumerate(result_list):
    annos = item['annotations']
    drop_list=[]
    print("annos:", annos)
    item['annotations'] = [x for x in annos if x['start']>=0]

#%%找到 result_list 中所有具有负起始位置的注释，然后打印包含这些注释的整个字典项，以便进行进一步的检查或处理
for idx,item in enumerate(result_list):
    annos = item['annotations']
    for a in annos:
        if a['start']<0:
            print(item)


for example in compare_list:
    for annotation in example['annotations']:
        #We expect the key of label to be label but the data has tag
        annotation['label'] = annotation['tag']

for example in result_list:
    for annotation in example['annotations']:
        #We expect the key of label to be label but the data has tag
        annotation['label'] = annotation['tag']
    with open('../data/output/4o-open_ai_EFSA_0_911_result.json', 'wt', encoding='utf8') as f:
        print(json.dumps(result_list, indent=2, ensure_ascii=False), file=f)

#计算得分
from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import TrainingDatasetCRF
from sequence_aligner.containers import TraingingBatch
from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-pretrain')
tokenizer = BertTokenizerFast.from_pretrained('finbert')
label_set = LabelSet(labels=["中立", "正面", "负面"])  # label in this dataset

dataset = TrainingDatasetCRF(data=compare_list, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
dataset_openai = TrainingDatasetCRF(data=result_list, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
#%%
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
# from util.process import ids_to_labels,Metrics,Metrics_e
from seqeval.scheme import BILOU

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

    if i==0:
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

report=classification_report(label_list, pred_label_list, mode='strict', scheme=BILOU)
print(report)