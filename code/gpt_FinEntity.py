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
    raw = json.load(open('../data/input/FinEntity.json'))
    raw = raw[:num]
    # openai.api_key = "sk-oGppXnowHQwOBrwL93945dDe99124523B240Da28061e78Ad"
    # openai.api_base = "https://www.jcapikey.com/v1"

    openai.api_key = "sk-uOXJMAbGOudoa6XJ703eE766B21b49E6Be4c245bE418D356"
    openai.api_base = "https://api.expansion.chat/v1"

    sys_prompt = "Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. "

    user_prompt = """
            Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into ‘Neutral’, ‘Positive’, or ‘Negative’ classes. 
            Considering every sentence as a String in python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing.
            Do not give explanations for the sentiment. In the output,Tag means sentiment; value means entity name. If no entity is found in the sentence, the response should be empty. 
            Output quadruple format example: {"start": 0, "end": 7, "value": "Kellogg", "tag": "Neutral"}. Use line breaks to separate different quadruples.
The sentence may contain different numbers of financial entities. Please pay attention to the output format and do not output general and irrelevant content such as "customer" and "company".
            """
    # user_prompt = """
    #     Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into ‘Neutral’, ‘Positive’, or ‘Negative’ classes.
    #     Considering every sentence as a String in python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing.
    #     Do not give explanations for the sentiment. In the output,Tag means sentiment; value means entity name. If no entity is found in the sentence, the response should be empty.
    #     The sentence: "Other U.S. companies have made similar moves, including social media site Reddit Inc and Mobileye, the self-driving car unit of Intel Corp <INTC.O>. "
    #     """
    assist_prompt = """{"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}\n{"start": 128, "end": 138, "value": "Intel Corp", "tag": "Neutral"}
        """
    user_prompt2 = """Kellogg <K.N>, however, based the corporate headquarters for its largest business, snacks, in Chicago after announcing a split into three independent companies this summer. [nL4N2Y822D]
        """
    assist_prompt2 = """{"start": 0, "end": 7, "value": "Kellogg", "tag": "Neutral"}
        """

    user_prompt3 = """Rival Oracle <ORCL.N> says in a statement on its website it has withdrawn all products, services and support for Russian and Belarusian companies, subsidiaries and partners. An Oracle spokesperson declined further comment.
        """

    assist_prompt3 = """{"start": 183, "end": 177, "value": "Oracle", "tag": "Neutral"}\n{"start": 6, "end": 12, "value": "Oracle", "tag": "Neutral"}
        """

    result_list = []
    compare_list = []
    for item in raw:
        sentence = item['content']
        print(sentence)
        # sentence ='Nearly all major S&P 500 sectors are red, with materials <.SPLRCM> and communications services <.SPLRCL> taking the biggest hits. Staples <.SPLRCS> and healthcare <.SPXHC> are posting small gains.'
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4o",
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
                model="gpt-3.5-turbo",
                # model="gpt-4o",
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
    with open('../data/2LW_output/1.0prompt_new_api/3.5_new_api_FinEntity_1226_2.json', 'wt') as f:
        print(json.dumps(result_list,indent=2), file=f)

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
                "tag": "Positive"
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
    with open('../data/2LW_output/1.0prompt_new_api/3.5_new_api_FinEntity_1226_sort_2.json', 'wt') as f:
        print(json.dumps(result_list,indent=2), file=f)
    print("COMPLETE")

    #%%
    print(len(compare_list))
    print(len(result_list))
    return compare_list, result_list


def get_compare(num):
    raw = json.load(open('../data/input/FinEntity.json'))
    raw = raw[:num]
    compare_list = []
    for item in raw:
        compare_list.append(item)
    result_list = json.load(open('../data/output/open_ai_FinEntity.json'))
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
    with open('../data/output/open_ai_FinEntity_resultAAAAAAAAAAA.json', 'wt') as f:
        print(json.dumps(result_list, indent=2), file=f)
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



for example in result_list:
    for annotation in example['annotations']:
        #We expect the key of label to be label but the data has tag
        annotation['label'] = annotation['tag']


#计算得分
from sequence_aligner.labelset import LabelSet
from sequence_aligner.dataset import TrainingDatasetCRF
from sequence_aligner.containers import TraingingBatch
from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-pretrain')
tokenizer = BertTokenizerFast.from_pretrained('finbert')
label_set = LabelSet(labels=["Neutral", "Positive", "Negative"])  # label in this dataset

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