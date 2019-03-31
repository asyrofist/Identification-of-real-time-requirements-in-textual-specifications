import random
from decimal import *
import numpy as np


class Fetch(object):
    @staticmethod
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


    @staticmethod
    def get_total_text(doc_list, keywords):
        """
            @param doc_list: list of filename []
            @return total_text: all sentences in these doc
                type: list of string [string, string, ...] -> [sentence, sentence, ...]
        """
        total_list = []
        for doc in doc_list:
            with open(r"/src/data/" + str(doc), "rt", encoding="utf-8") as f:
                total_list += [sentence.split('\n')[0] for sentence in f.readlines() if Fetch.has_keyword(sentence, keywords)]
        return total_list




class ChangeRatio:
    @staticmethod
    def treat(origin):
        """
            @param origin: list (of string)
            @return: {sentence : type}
                type:  map, {string : int}
        """
        map_to_type = [sentence.split("😂") for sentence in origin]
        return {m[0]: int(m[1]) for m in map_to_type}


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


    @staticmethod
    def adjust_sub(now_ratio, ratio, now_text, base=0, the_other=1):
        """
            @param now_ratio: [int, int], target text's ratio
            @param ration: [int, int], target ratio
            @param now_text: [[string,..], [string,..]], target text
            @param base_index, adjust_index: int, used to compare strings' ratio in this two index
            @return: nothing
                    adjust ratio in subsample mode
        """
        if ChangeRatio.ratio_low(now_ratio, ratio, base, the_other):
            random.shuffle(now_text[base])
            now_text[base] = now_text[base][:int(len(now_text[the_other]) * ratio[base] / ratio[the_other])]
        elif ChangeRatio.ratio_high(now_ratio, ratio, base, the_other):
            random.shuffle(now_text[the_other])
            now_text[the_other] = now_text[the_other][:int(len(now_text[base]) * ratio[the_other] / ratio[base])]
        return now_text
        

    @staticmethod
    def adjust_over(now_ratio, ratio, now_text, base=0, the_other=1):
        """
            @param now_ratio: [int, int], target text's ratio
            @param ration: [int, int], target ratio
            @param now_text: [[string,..], [string,..]], target text
            @param base_index, adjust_index: int, used to compare strings' ratio in this two index
            @return: nothing
                    adjust ratio in oversample mode
        """
        if ChangeRatio.ratio_low(now_ratio, ratio, base, the_other):
            random.shuffle(now_text[the_other])
            now_text[the_other] += now_text[the_other][:int(len(now_text[base]) * ratio[the_other] / ratio[base])]
        elif ChangeRatio.ratio_high(now_ratio, ratio, base, the_other):
            random.shuffle(now_text[base])
            now_text[base] += now_text[base][:int(len(now_text[the_other]) * ratio[base] / ratio[the_other])]
