import pandas as pd

PREPROCESSING = [
    'stemming',
    'lemma',
    'tokenization',
    'spelling'
]

def get_preprocessed_data(data, P_type):
    """

    :param data: Dataset in format "id","qid1","qid2","question1","question2","is_duplicate"
    :param P_type: str, one of PREPROCESSING literal.
    :return: transformed dataset
    """

    data = data.apply(stemming, axis=1)



def stemming(row):
    """

    :param row:
    :return:
    """
    return row
