from .utils import *


def fetch(number_of_sentences: int, doc_list: tuple = tuple(range(1, 49)), keywords: list = None, seed=0):
    """
        fetch sentences from doc_list. (might with keywords)

        :param number_of_sentences:
        :param doc_list: list of int, [1,2,3, ...]
        :param keywords: list of string, [keyword1, keyword2, ...], the keyword used to select sentence
        :param seed: rand seed
        :return: list of sentences, which are selected from the doc_list
            type: list of string [string, string, ...] -> [sentence, sentence, ...]
    """
    keywords = [str()] if keywords is None else keywords
    text = Fetcher.get_total_text(doc_list, keywords)
    random.Random(seed).shuffle(text)

    return text[:number_of_sentences]


def change_inner_ratio(x, label, mode, rand_seed, ratio):
    pos_label = (0, 1)
    neg_label = (1, 0)
    data = list(zip(x, label))
    pos, neg = [], []
    for v, lab in data:
        if lab == pos_label:
            pos.append((v, lab))
        elif lab == neg_label:
            neg.append((v, lab))
        else:
            raise ValueError('Unknown label', lab)
    pos, neg = change_feature_ratio(pos, neg, ratio, mode=mode, seed=rand_seed)
    data = pos + neg
    random.Random(0).shuffle(data)
    x, label = list(zip(*data))
    return x, label


def change_ratio(origin, division, ratio, mode, seed=0):
    """
        change the ratio of difference type of sentences

        :param origin: list (of string), format is sentence😂type😂description
        :param ratio: list, [elem_1, elem_2]
        :param division: list of list, [[], []]
        :param mode: string, "subSample" or "oversample"
        :return: text that suitable for the :param ratio
            type: same as origin
    """
    # text : map, from sentences to 0/1/2
    text = RatioChanger.get_map(origin)

    # [[strings in type0], [strings in type1], [strings in type2]]
    typed_text = RatioChanger.partition(text)

    part1 = []
    part2 = []
    for i in division[0]:
        part1.extend(typed_text[i])
    for i in division[1]:
        part2.extend(typed_text[i])

    now_ratio = [len(part1), len(part2)]
    now_text = [part1, part2]

    base = 0
    the_other = 1

    if mode == "subSample":
        RatioChanger.adjust_sub(now_ratio, ratio, now_text, base, the_other, seed)
    elif mode == "overSample":
        RatioChanger.adjust_over(now_ratio, ratio, now_text, base, the_other, seed)

    now_text = now_text[0] + now_text[1]
    text_0 = [sentence for sentence in now_text if text[sentence] == 0]
    text_1 = [sentence for sentence in now_text if text[sentence] == 1]
    text_2 = [sentence for sentence in now_text if text[sentence] == 2]
    return text_0, text_1, text_2


def change_feature_ratio(pos, neg, ratio, mode, seed=0):
    pos = pos.copy()
    neg = neg.copy()
    now_ratio = [len(pos), len(neg)]
    now_text = [pos, neg]
    base = 0
    the_other = 1

    if mode == "subSample":
        RatioChanger.adjust_sub(now_ratio, ratio, now_text, base, the_other, seed)
    elif mode == "overSample":
        RatioChanger.adjust_over(now_ratio, ratio, now_text, base, the_other, seed)

    return now_text[0], now_text[1]


def test_fetch():
    print(fetch(10, (1, 2)))


def test_change_ratio():
    text = [
        "1_1😂1😂", "2_1😂1😂", "3_1😂1😂", "4_1😂1😂", "5_1😂1😂", "6_1😂1😂", "7_1😂1😂", "8_1😂1😂", "9_1😂1😂",
        "10_1😂1😂",
        "1_0😂0😂", "2_0😂0😂", "3_0😂0😂", "4_0😂0😂", "5_0😂0😂", "6_0😂0😂", "7_0😂0😂", "8_0😂0😂", "9_0😂0😂",
        "10_0😂0😂",
        "1_2😂2😂", "2_2😂2😂", "3_2😂2😂", "4_2😂2😂", "5_2😂2😂", "6_2😂2😂", "7_2😂2😂", "8_2😂2😂", "9_2😂2😂",
        "10_2😂2😂",
    ]
    print(change_ratio(text, [[0, 1], [2]], [13, 4], "overSample"))
    print(change_ratio(text, [[0, 1], [2]], [13, 17], "subSample"))


if __name__ == "__main__":
    # test_fetch()
    test_change_ratio()
