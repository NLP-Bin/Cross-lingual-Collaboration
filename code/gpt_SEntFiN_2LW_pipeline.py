import copy
import os
import re
from datetime import datetime

import openai
import json
import time

#计算得分
from tqdm import tqdm

from Finentiity_gpt.code.sequence_aligner.labelset import LabelSet
from Finentiity_gpt.code.sequence_aligner.dataset import TrainingDatasetCRF
from Finentiity_gpt.code.sequence_aligner.containers import TraingingBatch
from transformers import BertTokenizerFast

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
# from util.process import ids_to_labels,Metrics,Metrics_e
from seqeval.scheme import BILOU
# %%
def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list


def gpt(start, num, model_choice, to_language):
    raw = json.load(open('../data/input/SEntFiN_test.json'))
    raw = raw[start:num]

    # openai.api_key = "sk-oGppXnowHQwOBrwL93945dDe99124523B240Da28061e78Ad"
    # openai.api_base = "https://www.jcapikey.com/v1"

    openai.api_key = "sk-uOXJMAbGOudoa6XJ703eE766B21b49E6Be4c245bE418D356"
    openai.api_base = "https://api.expansion.chat/v1"

    sys_prompt = "Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. "


#     EFSA_prompt_V3 = """Identify entities in the content that belong to companies or organizations and classify their corresponding sentiment into ‘neutral,’ ‘positive,’ or ‘negative.’. Stock codes of companies do not need to be marked.
# Consider each sentence as a string in Python, and provide the start and end indices (zero-based indexing) to mark the boundaries of the entities, including spaces and punctuation.
# Do not provide any explanation for the sentiment classification. In the output, “Tag” represents the sentiment, and “value” represents the entity name. If no entity is found in a sentence, the output should be empty.
# Output format example: {"start": 0, "end": 7, "value": "Kellogg", "tag": "neutral"}.
# Use line breaks to separate different quadruples.
#
# The sentence may contain varying numbers of financial entities. Do not mark general or irrelevant content such as "customer" or "company" as well as countries, personal names, dates, and job titles.
# Please pay attention to the output format.
#
# TEXT:"""
    EFSA_prompt_V3 = """Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into ‘neutral’, ‘positive’, or ‘negative’ classes.
Considering every sentence as a String in python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing.
Do not give explanations for the sentiment. In the output,Tag means sentiment; value means entity name.
The output format strictly follows the followingform: {\"value\": \"power sector\", \"start\": 23, \"end\": 35, \"tag\": \"negative\"}\n{\"value\": \"Equinomics\", \"start\": 82, \"end\": 92, \"tag\": \"neutral\"}
TEXT:"""


    Translation_EFSA_prompt_V3 = f"""\nTranslate the financial text accurately into {to_language}, reconsider it from the {to_language} perspective, and perform a more comprehensive entity-level sentiment analysis. 
    Consider entities such as organizations, companies, products, brands, etc. Filter out meaningless generic terms.
    Provide the translation and the new quadruple output in the following format:
    Translation: 
    Output: """
    #这个结合V3+V3+V3可以

    com_a1_a2_prompt_v3 = f"""Please refer to the {to_language} result to revise the English result, and provide the final reasonable English result. 
    Ensure that an appropriate sentiment polarity is provided for each occurrence of an entity in the English text. 
    Only output the result, do not output any irrelevant content."""
    #


    save_path = f'../data/2LW_SEntFiN_output/{to_language}/2LW_pipeline_{model_choice}_SEntFiN_{to_language}_0shot_test+NUM'

    result_list = []
    compare_list = []
    try:
        # 使用 enumerate 获取索引和元素
        for index_all, item in enumerate(tqdm(raw, desc="Processing items")):

            result_dict = {}
            sentence = item['content']
            result_dict['content'] = sentence
            result_dict['truth_annotations'] = item['annotations']

            # prompt_1 = EFSA_prompt+sentence
            prompt_1 = EFSA_prompt_V3 + sentence
            # prompt_1 = EFSA_prompt
            # print("\n原语言执行EFSA任务prompt_1: ", prompt_1)
            # sentence ='Nearly all major S&P 500 sectors are red, with materials <.SPLRCM> and communications services <.SPLRCL> taking the biggest hits. Staples <.SPLRCS> and healthcare <.SPXHC> are posting small gains.'
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_1},
                # {"role": "user", "content": sentence},
            ]
            # print(model_choice)
            try:
                rsp = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=messages,
                    temperature=0.0,
                )
            except:  #未获取到结果，再次获取
                result_dict['annotations_answer1'] = []
                print("retry request1")
                time.sleep(2)
                rsp = openai.ChatCompletion.create(
                    model= model_choice,
                    messages=[{"role": "user", "content": sentence}, ], temperature=0.0, )
            else:   #成功获取到结果



                # 结果存入answer1
                answer1 = rsp['choices'][0]['message']['content']
                # print("\n原语言执行结果answer1:", answer1)

                # 处理获取到的第一次的结果################################

                answer1 = answer1.strip()
                answer1_list = answer1.split('\n')
                # 以\n区分不同的实体组  {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}\n{"start": 128, "end": 138, "value": "Intel Corp", "tag": "Neutral"}
                anno_list = []
                # 如果message['content']为空
                if len(answer1_list) == 0:
                    result_dict['annotations_answer1'] = []

                else:
                    for res in answer1_list:  # 对于每一个实体组 {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}
                        index_left = res.find('{')
                        index_right = res.find('}')
                        if index_right == -1 or index_left == -1:
                            continue
                        res = res[index_left:index_right + 1]
                        sub_json = json.loads(res)
                        anno_list.append(sub_json)
                    result_dict['annotations_answer1'] = anno_list




            ########    step 2


            prompt2 = Translation_EFSA_prompt_V3
            messages.append({"role": "assistant", "content": answer1})
            messages.append({"role": "user", "content": prompt2})
            # print("\n\nmessages:\n\n", messages)
            try:
                rsp2 = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=messages,
                    temperature=0.0,
                )




            except:  # 未获取到结果，再次获取
                print("retry request2")
                time.sleep(2)
                rsp2 = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": prompt2}], temperature=0.0, )
            else:  # 成功获取到结果
                respond2 = rsp2['choices'][0]['message']['content']
                # print("\n转换语言及其执行结果:\n", respond2)

                # 正则提取 Translation 和 Output 之间的内容
                Translation_match = re.search(r"Translation:\s*(.*?)\s*Output:", respond2, re.S)
                # 如果找到，去除多余空白
                if Translation_match:
                    translation_content = Translation_match.group(1).strip()
                    # result_dict['translation_content'] = bytes(translation_content, "utf-8").decode("unicode_escape")
                    result_dict['translation_content'] = translation_content

                    # print(translation_content)
                else:
                    result_dict['translation_content'] = ""
                    print("Content between 'Translation:' and 'Output:' not found.")

                # 匹配 Output 后的内容
                output_match = re.search(r"Output:\s*(.*)", respond2, re.DOTALL)
                if output_match:
                    answer2 = output_match.group(1).strip()  # 获取 Output 后的内容并去掉多余空白
                    if answer2:
                        # result_dict['translation_answer2'] = bytes(answer2, "utf-8").decode("unicode_escape")
                        result_dict['translation_answer2'] = answer2
                        # print("\nanswers2:\n", answer2)
                    else:
                        result_dict['translation_answer2'] = ""
                        print("Output 后没有内容")
                else:
                    result_dict['translation_answer2'] = ""
                    print("未找到 Output 标记")



            ########    step 3

            prompt3 = com_a1_a2_prompt_v3
            messages.append({"role": "assistant", "content": respond2})
            messages.append({"role": "user", "content": prompt3})
            # print("\n\nmessages:\n\n", messages)
            try:
                rsp3 = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=messages,
                    temperature=0.0,
                )



            except:  # 未获取到结果，再次获取
                print("retry request3")
                time.sleep(2)
                rsp3 = openai.ChatCompletion.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": sentence}], temperature=0.0, )
            else:  # 成功获取到结果

                # 如果获取结果为空,直接跳入下一条数据的循环
                if len(rsp3['choices']) == 0:
                    result_dict['annotations'] = []
                    result_list.append(result_dict)
                    compare_list.append(item)
                    continue
                # 如果获取结果不为空，结果存入answer2
                respond3 = rsp3['choices'][0]['message']['content']
                print("\n合并后的最终结果:\n", respond3)

                #处理获取到的最终结果################################

                respond3 = respond3.strip()
                answer_end = respond3.split('\n')
                #以\n区分不同的实体组  {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}\n{"start": 128, "end": 138, "value": "Intel Corp", "tag": "Neutral"}
                anno_list = []
                #如果message['content']为空
                if len(answer_end) == 0:
                    result_dict['annotations'] = []
                    result_list.append(result_dict)
                    compare_list.append(item)
                    continue
                for res in answer_end:  #对于每一个实体组 {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}
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

    except KeyboardInterrupt:
        print("\nProcessing interrupted manually.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        with open(save_path+str(num)+'.json', 'wt', encoding='utf-8') as f:
            print(json.dumps(result_list, indent=2, ensure_ascii=False), file=f)
        # 将报告写入日志文件
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n进程结束，执行至第 {index_all} 条,\n 内容是\n {item} \n")
            log_file.write("\n\n")  # 添加换行方便阅读

    result_list = sort_sort(result_list, answer_choice="annotations_answer1") # 对第一步直接生成的结果的每个元素进行处理，修正起始和结束标签。
    result_list = sort_sort(result_list, answer_choice="annotations")  # 对 最终结果中的每个元素进行处理，修正起始和结束标签。

    with open(save_path+str(num)+'sort.json', 'wt', encoding='utf-8') as f:
        print(json.dumps(result_list, indent=2, ensure_ascii=False), file=f)
    print("COMPLETE")


    print(len(compare_list))
    print(len(result_list))
    return compare_list, result_list

def remove_empty_dicts(lst):
    remove_empty_result = []
    for item in lst:
        if item!= {}:  # 检查元素是否不为空字典
            remove_empty_result.append(item)
    return remove_empty_result

def sort_sort(result_list, answer_choice):
    # Correcting start and end tags    对 result_list 中的每个元素进行处理，修正起始和结束标签。
    # 代码确保了注释在文本中的位置不重叠，并且更新了注释的起始位置和结束位置
    for i, item in enumerate(result_list):
        text = item['content']
        annos = item[answer_choice]  # {}{}
        if (annos == [{}]):
            item[answer_choice] = [{
                "start": 0,
                "end": 0,
                "value": "aaaaa",
                "tag": "positive"
            }]
            sorted_annos = item[answer_choice]
        else:
            try:
                sorted_annos = sorted(annos, key=lambda x: x['start'])  # 按start值对每组排序
            except:
                print("结果中中含有空字典：\n", item['content'], "\n", annos)
                remove_empty_annos = remove_empty_dicts(annos)
                item[answer_choice] = remove_empty_annos
                print("去除空字典后的结果：\n", remove_empty_annos)
                sorted_annos = sorted(remove_empty_annos, key=lambda x: x['start'])  # 按start值对每组排序
        value_list = []
        start_list = []
        drop_list = []
        for indx, sub_annos in enumerate(
                sorted_annos):  # 对于每组{"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}

            try:
                value = sub_annos['value']  # "value": "Reddit Inc"
            except:
                print("", "结果中缺value值：\n", item['content'], "\n", annos)

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
    return result_list




def evaluate(compare_list, result_list1, answer_choice, log_file_path):
    # %%只保留起始位置大于 0 的组
    for idx, item in enumerate(result_list1):
        annos = item[answer_choice]
        drop_list = []
        # print("annos:", annos)
        try:
            item[answer_choice] = [x for x in annos if x['start'] >= 0]
        except:
            print("", "结果中缺start值：\n", item['content'], "\n", annos)

    # %%找到 result_list 中所有具有负起始位置的注释，然后打印包含这些注释的整个字典项，以便进行进一步的检查或处理
    for idx, item in enumerate(result_list1):
        annos = item[answer_choice]
        for a in annos:
            if a['start'] < 0:
                print(item)

    for example in compare_list:
        for annotation in example['annotations']:
            # We expect the key of label to be label but the data has tag
            if 'tag' in annotation:
                annotation['label'] = annotation['tag']
            else:
                annotation['label'] = "positive"

    for example in result_list1:

        for annotation in example[answer_choice]:
            # We expect the key of label to be label but the data has tag
            annotation['label'] = annotation['tag']
        if(answer_choice == "annotations_answer1"):
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            example["annotations"] = example[answer_choice]   #需要将example[answer_choice]赋给"annotations"才能计算
    # tokenizer = BertTokenizerFast.from_pretrained('yiyanghkust/finbert-pretrain')
    tokenizer = BertTokenizerFast.from_pretrained('finbert')
    label_set = LabelSet(labels=["neutral", "positive", "negative"])  # label in this dataset

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
    print(answer_choice, "Classification Report:\n", report)


    # 将报告写入日志文件
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{answer_choice} Classification Report:\n")
        log_file.write(report)
        log_file.write("\n\n")  # 添加换行方便阅读

    print(f"Report saved to {log_file_path}")

def get_compare_from_file(num):
    raw = json.load(open('../data/2LW_SEntFiN_output/French/2LW_pipeline_gpt-3.5-turbo_SEntFiN_French_0shot_test+NUM100.json', 'r', encoding='utf-8'))
    raw = raw[:num]
    compare_list = []
    for item in raw:
        compare_list.append(item)
    result_list = json.load(open('../data/2LW_output/2LW_3.5-open_ai_FinEntity_0shot_sort.json', 'r', encoding='utf-8'))[:num]
    print(len(compare_list))
    print(len(result_list))

    return compare_list, result_list



# model_choice = "gpt-3.5-turbo"   #  提示¥0.00175/K tokens    补全¥0.00525/K tokens
# model_choice="gpt-4o"          #  提示¥0.00875/K tokens    补全¥0.035/K tokens
model_choice="deepseek-chat"    #  提示¥0.0014/K tokens     补全¥0.0028/K tokens
#model_choice="chatglm_turbo"    #  提示¥0.014/K tokens      补全¥0.014/K tokens
#model_choice=""
#model_choice=""



# to_language = "French"
# to_language = "Spanish"
# to_language = "Russian"
# to_language = "Chinese"
# to_language = "Tibetan"
# to_language = "Arabic"
# to_language_list = ["French", "Spanish", "Russian", "Chinese", "Tibetan", "Arabic"]
to_language_list = ["Tibetan"]
for to_language in to_language_list:
    print("现在的辅助语言是", to_language, "\n")

    start = 0 # 从第0条开始，如果中断了，可以从那一条继续，节省资源。
    num = 100# 需完成数量的总条目

    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 拼接日志文件名
    log_file_name = f"log_{to_language}_{model_choice}_{current_time}+NUM{num}.log"
    log_file_path = f"../data/2LW_SEntFiN_output/{to_language}/{log_file_name}"   # 日志文件路径


    compare_list, result_list = gpt(start, num, model_choice, to_language)
    # compare_list, result_list = get_compare_from_file(num)

    # result_list2 = copy.deepcopy(result_list)
    # result_list2 = result_list


    answer_choice = "annotations"
    # print(result_list[0][answer_choice])
    evaluate(compare_list, result_list, answer_choice, log_file_path)



    answer_choice = "annotations_answer1"
    # print(result_list[0][answer_choice])
    evaluate(compare_list, result_list, answer_choice, log_file_path)




