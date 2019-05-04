import random
from decimal import *
import os


class Fetcher(object):
    @staticmethod
    def has_keyword(sentence: str, keywords: list):
        """
        if sentence has any keyword in keywords

        :param sentence: string
        :param keywords: list of string, [string, string, ...]
        :return : boolean
        """
        for key in keywords:
            if key in sentence:
                return True
        return False

    @staticmethod
    def get_total_text(doc_list, keywords):
        """
        get all sentences in these docs

        :param doc_list: list of filename []
        :param keywords: keywords
        :return: list of string [string, string, ...] -> [sentence, sentence, ...]
        """
        total_list = []

        def is_capable(filename: str, type_list: list):
            filename = filename.lower()
            for t in type_list:
                t = t.lower()
                if filename.endswith(t):
                    return True
            return False

        def get_doc_index(path):
            # print(path)
            filename = os.path.basename(path)
            index = filename.split('#')[0]
            try:
                index = int(index)
                return index
            except ValueError:
                return -1

        root = r'../data/doc_set'
        filenames = [os.path.join(dp, f) for dp, _, filenames in os.walk(root)
                     for f in filenames if is_capable(f, ['txt'])]
        filenames = [file for file in filenames if get_doc_index(file) in doc_list]

        for name in filenames:
            with open(name, "rt", encoding="utf-8") as f:
                total_list += [sentence.strip() for sentence in f.readlines() if
                               Fetcher.has_keyword(sentence, keywords)]
        total_list = [sentence.lower() for sentence in total_list if len(sentence)]
        return total_list


class RatioChanger:
    @staticmethod
    def get_map(origin):
        """
        map every sentence to type

        :param origin: list (of string)
        :return: {sentence : type}
            type:  map, {string : int}
        """
        # print(origin[0].split("😂"))
        map_to_type = [sentence.split("😂") for sentence in origin]
        return [(m[3], int(m[1])) for m in map_to_type if not m[2]]

    @staticmethod
    def partition(text):
        """
        typed sentence

        :param text: map, {string : int} == {sentence : type}
        :return: list of list
        """
        return [
            [key for key, type in text if type == 0],
            [key for key, type in text if type == 1],
            [key for key, type in text if type == 2]
        ]

    @staticmethod
    def ratio_low(now_ratio, ratio, base_index, adjust_index):
        """
        if now_ratio is lower than target ratio

        :param now_ratio: [int, int], target text's ratio
        :param ratio: [int, int], target ratio
        :param base_index: int, used to compare strings' ratio in this two index
        :param adjust_index:
        :return: boolean
        """
        return Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) - Decimal(
            ratio[adjust_index]) / Decimal(ratio[base_index]) < Decimal("0")

    @staticmethod
    def ratio_high(now_ratio, ratio, base_index, adjust_index):
        """
        if now_ratio is higher than target ratio

        :param now_ratio: [int, int], target text's ratio
        :param ratio: [int, int], target ratio
        :param base_index: int, used to compare strings' ratio in this two index
        :param adjust_index:
        :return: boolean
        """
        return Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) - Decimal(
            ratio[adjust_index]) / Decimal(ratio[base_index]) > Decimal("0")

    @staticmethod
    def ratio_equal(now_ratio, ratio, base_index, adjust_index):
        """
        if now_ratio is equal to target ratio

        :param now_ratio: [int, int], target text's ratio
        :param ratio: [int, int], target ratio
        :param base_index: int, used to compare strings' ratio in this two index
        :param adjust_index:
        :return: boolean
        """
        return (Decimal(now_ratio[adjust_index]) / Decimal(now_ratio[base_index]) ==
                Decimal(ratio[adjust_index]) / Decimal(ratio[base_index]))

    @staticmethod
    def adjust_sub(now_ratio, ratio, now_text, base=0, the_other=1, seed=0):
        """
        adjust ratio in subsample mode
        :param now_ratio: [int, int], target text's ratio
        :param ratio: [int, int], target ratio
        :param now_text: [[string,..], [string,..]], target text
        :param base:
        :param the_other: int, used to compare strings' ratio in this two index
        :return: None
        """
        if RatioChanger.ratio_low(now_ratio, ratio, base, the_other):
            random.Random(seed).shuffle(now_text[base])
            new_size = len(now_text[the_other]) * ratio[base] // ratio[the_other]
            now_text[base] = now_text[base][:new_size]

        elif RatioChanger.ratio_high(now_ratio, ratio, base, the_other):
            random.Random(seed).shuffle(now_text[the_other])
            new_size = len(now_text[base]) * ratio[the_other] // ratio[base]
            now_text[the_other] = now_text[the_other][:new_size]

    @staticmethod
    def adjust_over(now_ratio, ratio, now_text, base=0, the_other=1, seed=0):
        """
        adjust ratio in oversample mode
        :param now_ratio: [int, int], target text's ratio
        :param ratio: [int, int], target ratio
        :param now_text: [[string,..], [string,..]], target text
        :param base:
        :param seed
        :param the_other: int, used to compare strings' ratio in this two index
        :return: None
        """
        if RatioChanger.ratio_low(now_ratio, ratio, base, the_other):
            random.Random(seed).shuffle(now_text[the_other])
            extend = len(now_text[base]) * ratio[the_other] // ratio[base]
            leng = len(now_text[the_other])
            now_text[the_other] *= extend // leng
            now_text[the_other] += now_text[the_other][:extend % leng]

        elif RatioChanger.ratio_high(now_ratio, ratio, base, the_other):
            random.Random(seed).shuffle(now_text[base])
            extend = len(now_text[the_other]) * ratio[base] // ratio[the_other]
            leng = len(now_text[base])
            now_text[base] *= extend // leng
            now_text[base] += now_text[base][:extend % leng]
