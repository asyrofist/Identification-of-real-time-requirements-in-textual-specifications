import random
from decimal import *
import numpy as np
from utils import *

def fetch(numberOfAllSentences, doc_list=range(1, 49), keywords=[""]):
    """
        @param numberOfAllSentences:
        @param doc_list: list of int, [1,2,3, ...]
        @param keywords: list of string, [keyword1, keyword2, ...], the keyword used to select sentence
        @return: list of sentences, which are selected from the doc_list
            type: list of string [string, string, ...] -> [sentence, sentence, ...]
    """
    text = Fetch.get_total_text(doc_list, keywords)
    random.shuffle(text)

    return text[:numberOfAllSentences]


def change_ratio(origin, division, ratio, mode):
    """
        @param origin: list (of string), format is sentence😂type😂description
        @param ratio: list, [elem_1, elem_2]
        @param division: list of list, [[], []]
        @param mode: string, "subSample" or "oversample"
        @return: text that suitable for the @param ratio
            type: same as origin
    """
    text = ChangeRatio.treat(origin)  # text : map, from sentences to 0/1/2
    typed_text = ChangeRatio.partition(text)  # [[string...(type0)], [string...(type1)], [string...(type2)]]
    part1 = []
    part2 = []
    for i in division[0]:
        part1 += typed_text[i]
    for i in division[1]:
        part2 += typed_text[i]
    now_text = [part1, part2]
    now_ratio = [len(now_text[0]), len(now_text[1])]

    base = 0
    the_other = 1

    if mode == "subSample":
        ChangeRatio.adjust_sub(now_ratio, ratio, now_text, base, the_other)
    elif mode == "overSample":
        ChangeRatio.adjust_over(now_ratio, ratio, now_text, base, the_other)

    return now_text



def test_fetch():
    print(fetch(10, [1, 2]))

def test_change_ratio():
    text = [
        "1_1😂1😂", "2_1😂1😂", "3_1😂1😂", "4_1😂1😂", "5_1😂1😂", "6_1😂1😂", "7_1😂1😂", "8_1😂1😂", "9_1😂1😂", "10_1😂1😂",
        "1_0😂0😂", "2_0😂0😂", "3_0😂0😂", "4_0😂0😂", "5_0😂0😂", "6_0😂0😂", "7_0😂0😂", "8_0😂0😂", "9_0😂0😂", "10_0😂0😂",
        "1_2😂2😂", "2_2😂2😂", "3_2😂2😂", "4_2😂2😂", "5_2😂2😂", "6_2😂2😂", "7_2😂2😂", "8_2😂2😂", "9_2😂2😂", "10_2😂2😂",
    ]
    print(change_ratio(text, [[0, 1], [2]], [4, 1], "overSample"))
    print(change_ratio(text, [[0, 1], [2]], [4, 1], "subSample"))


if __name__ == "__main__":
    test_fetch()
    test_change_ratio()
