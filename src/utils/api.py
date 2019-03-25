import random
from decimal import *
import numpy as np

############## fetch begin ###############

"""
    TODO note: the doc must be save in directory "/files"
"""

def has_keyword(sentence, keywords):
    """
        @param sentence: string
        @param keywords: list of string, [string, string, ...]
        @return : if sentence has any keyword in keywords
            type: boolean
    """
    for key in keywords:
        if sentence.find(key) != -1:
            return True
    return False

def get_total_text(doc_list, keywords):
    """
        @param doc_list: list of filename []
        @return total_text: all sentences in these doc
            type: list of string [string, string, ...] -> [sentence, sentence, ...]
    """
    total_list = []
    for doc in doc_list:
        with open("files/"+str(doc), "rt", encoding="utf-8") as f:
            total_list += [sentence.split('\n')[0] for sentence in f.readlines() if has_keyword(sentence, keywords)]
    return total_list


def fetch(numberOfAllSentences, doc_list=[i for i in range(1, 49)], shuffle=False, keywords=[""]):
    """
        @param numberOfAllSentences: 
        @param doc_list: list of int, [1,2,3, ...]
        @param shuffle: boolean, true or false, decide whether shuffe the oeder of the sentences
        @param keywords: list of string, [keyword1, keyword2, ...], the keyword used to select sentence
        @return: list of sentences, which are selected from the doc_list 
            type: list of string [string, string, ...] -> [sentence, sentence, ...]
    """

    text = []
    total_text = get_total_text(doc_list, keywords)
    len_total_text = len(total_text)
    for i in range(numberOfAllSentences if numberOfAllSentences < len_total_text else len_total_text):
        sentence = random.choice(total_text)
        text += [sentence]
        total_text.remove(sentence)
    if shuffle:
        random.shuffle(text)

    return text


def test_fetch():
    print(fetch(10, [1,2]))



############## fetch end #################

########## change ratio begin ############


def update_now_ratio(now_text):
    return [len(now_text[i]) for i in range(2)]


def treat(origin):
    """
        @param origin: list (of string)
        @return: {sentence : type}
            type:  map, {string : int}
    """
    map = [sentence.split("😂") for sentence in origin]
    return {m[0]: int(m[1]) for m in map}


def partition(text):
    """
        @param text: map, {string : int} == {sentence : type}
        @return: typed sentence
            type:  list of list
    """
    return [
        [key for key in text.keys() if text[key] == 0],
        [key for key in text.keys() if text[key] == 1],
        [key for key in text.keys() if text[key] == 2]
    ]


def reduce_list_size(text, index):
    """
        @param text: [[string...(type0)], [string...(type1), ...]]
        @param index: int, means type
        @return: nothing
                remove a random list element
    """
    text[index].remove(random.choice(text[index]))


def add_list_size(now_text, typed_text, index):
    """
        @param now_text: [[string...(type0)], [string...(type1), ...]], our target text
        @param typed_text: [[string...(type0)], [string...(type1), ...]], typed text
        @param index: int, means type
        @return: nothing
                add a random element to now_text from typed_text
    """
    now_text[index].append(random.choice(typed_text[index]))


def ratio_low(now_ratio, ratio, base_index, adjust_index):
    """
        @param now_ratio: [int, int], target text's ratio
        @param ration: [int, int], target ratio
        @param base_index, adjust_index: int, used to compare strings' ratio in this two index
        @return if now_ratio is lower than target ratio
            type: boolean
    """
    return Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) - Decimal(
        ratio[adjust_index]) / Decimal(ratio[base_index]) < Decimal("0")


def ratio_high(now_ratio, ratio, base_index, adjust_index):
    """
        @param now_ratio: [int, int], target text's ratio
        @param ration: [int, int], target ratio
        @param base_index, adjust_index: int, used to compare strings' ratio in this two index
        @return if now_ratio is higher than target ratio
            type: boolean
    """
    return Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) - Decimal(
        ratio[adjust_index]) / Decimal(ratio[base_index]) > Decimal("0")


def ratio_equal(now_ratio, ratio, base_index, adjust_index):
    """
        @param now_ratio: [int, int], target text's ratio
        @param ration: [int, int], target ratio
        @param base_index, adjust_index: int, used to compare strings' ratio in this two index
        @return if now_ratio is equal than target ratio
            type: boolean
    """
    return (Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) == 
            Decimal(ratio[adjust_index]) / Decimal(ratio[base_index]))


def adjust_sub(now_ratio, ratio, now_text, typed_text, base_index=0, adjust_index=1):
    """
        @param now_ratio: [int, int], target text's ratio
        @param ration: [int, int], target ratio
        @param now_text: [[string,..], [string,..]], target text
        @param typed_text: [[string,..], [string,..], ...], typed text
        @param base_index, adjust_index: int, used to compare strings' ratio in this two index
        @return: nothing
                adjust ratio in subsample mode
    """
    if ratio_high(now_ratio, ratio, base_index, adjust_index):
        reduce_list_size(now_text, adjust_index)
    elif ratio_low(now_ratio, ratio, base_index, adjust_index):
        reduce_list_size(now_text, base_index)


def adjust_over(now_ratio, ratio, now_text, typed_text, base_index, adjust_index):
    """
        @param now_ratio: [int, int], target text's ratio
        @param ration: [int, int], target ratio
        @param now_text: [[string,..], [string,..]], target text
        @param typed_text: [[string,..], [string,..], ...], typed text
        @param base_index, adjust_index: int, used to compare strings' ratio in this two index
        @return: nothing
                adjust ratio in oversample mode
    """
    if ratio_low(now_ratio, ratio, base_index, adjust_index):
        add_list_size(now_text, typed_text, adjust_index)
    elif ratio_high(now_ratio, ratio, base_index, adjust_index):
        add_list_size(now_text, typed_text, base_index)


def change_ratio(origin, division ,ratio, mode):
    """
        @param origin: list (of string), format is sentence😂type😂description
        @param ratio: list, [elem_1, elem_2]
        @param division: list of list, [[], []]
        @param mode: string, "subSample" or "oversample"
        @return: text that suitable for the @param ratio
            type: same as origin
    """
    text = treat(origin)  # text : map, from sentences to 0/1/2
    typed_text = partition(text)  # [[string...(type0)], [string...(type1)], [string...(type2)]]
    part1 = []
    part2 = []
    for i in division[0]:
        part1 += [str for str in typed_text[i]]
    for i in division[1]:
        part2 += [str for str in typed_text[i]]
    now_text = [part1, part2]
    now_ratio = [len(now_text[0]), len(now_text[1])]
    if mode == "subSample":
        while not ratio_equal(now_ratio, ratio, 0, 1):
            adjust_sub(now_ratio, ratio, now_text, typed_text, 0, 1)
            now_ratio = update_now_ratio(now_text)
    elif mode == "overSample":
        while not ratio_equal(now_ratio, ratio, 0, 1):
            adjust_over(now_ratio, ratio, now_text, typed_text, 0, 1)
            now_ratio = update_now_ratio(now_text)
    return now_text


def test_change_ratio():
    text = {
        "1_1😂1😂", "2_1😂1😂", "3_1😂1😂", "4_1😂1😂", "5_1😂1😂", "6_1😂1😂", "7_1😂1😂", "8_1😂1😂", "9_1😂1😂", "10_1😂1😂",
        "1_0😂0😂", "2_0😂0😂", "3_0😂0😂", "4_0😂0😂", "5_0😂0😂", "6_0😂0😂", "7_0😂0😂", "8_0😂0😂", "9_0😂0😂", "10_0😂0😂",
        "1_2😂2😂", "2_2😂2😂", "3_2😂2😂", "4_2😂2😂", "5_2😂2😂", "6_2😂2😂", "7_2😂2😂", "8_2😂2😂", "9_2😂2😂", "10_2😂2😂",
    }
    res_text = change_ratio(text, [[0,1],[2]], [4, 1], "overSample")
    print(res_text)


############ change ratio end ############


if __name__ == "__main__":
    test_fetch()
    test_change_ratio()
