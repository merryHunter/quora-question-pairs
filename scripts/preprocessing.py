import pandas as pd
from nltk.stem.snowball import SnowballStemmer

PREPROCESSING = [
    # 'spelling',
    'stemming',
    'NO',

]

stemmer = SnowballStemmer("english")

def get_preprocessed_data(data, P_type):
    """

    :param data: Dataset in format "id","qid1","qid2","question1","question2","is_duplicate"
    :param P_type: str, one of PREPROCESSING literal.
    :return: transformed dataset
    """
    if P_type == 'stemming':
        data['question1'] = data['question1'].map(stemming)
        data['question2'] = data['question2'].map(stemming)
    elif P_type == 'spelling':
        pass

    return data


def stemming(s):
    """
    Snowball stemming.
    :param s:
    :return:
    """
    try:
        r = s.decode("utf-8")
        r = stemmer.stem(r)
        r = r.encode("ascii", "ignore")
    except:
        return s.encode("ascii", "ignore")
    return r


def spelling(s):
    
    return s