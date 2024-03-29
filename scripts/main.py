import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from preprocessing import get_preprocessed_data, PREPROCESSING
from models import train_xgb, predict_xgb
from sklearn.metrics import log_loss
from nltk.corpus import stopwords
import functools
from collections import defaultdict
from collections import Counter
import os.path
import editdistance
import time
import logging
from datetime import datetime
from sklearn.model_selection import KFold


timestamp = datetime.now().strftime('-%m-%d-%H:%M')
logFormatter = logging.Formatter("%(message)s")

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
fileHandler = logging.FileHandler("{0}/{1}.log".format('/home/ivan/', 'out' + str(timestamp)))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

RANDOM_STATE = 2017


# TODO: report what has been done, after do and report graph!
# TODO: add graph feature       5hr
# TODO: put edit_distance, graph leak features in __notebook__raw 3hr
# TODO: add LSI features

#not in priority
# TODO: add time elapsed for training / feature computing 0.5hr


words = None
counts = None
weights = None
q_dict = defaultdict(set)
stops = set(stopwords.words("english"))
TRAIN = pd.read_csv('../newdata/train.csv')
TEST = pd.read_csv('../newdata/test.csv')


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed


def conv(val):
    try:
        return np.float64(val)
    except:
        return np.float64(0)


def get_or_compute_features(data, dfile, feature_list, df_type):
    """
    Load dataset with specified features.
    :param dfile: filename of the .csv file from where it tries to load features listed in <feature_list>
    :param data: Dataset in format "id","qid1","qid2","question1","question2","is_duplicate"
    :param feature_list: list of features to be restored from file or computed in run-time on the dataset.
    :return: dataset with specified features
    """
    if os.path.isfile('../newdata/' + dfile):
        df = pd.read_csv('../newdata/' + dfile)
    else:
        df = pd.DataFrame()

    df['question1'] = data['question1'].copy()
    df['question2'] = data['question2'].copy()

    # if all features presented in the file, then no computing
    feats_to_compute = set(feature_list) - set(list(df))
    if len(feats_to_compute) == 0:
        return df

    compute_weights_count(data)

    feats_to_compute = set(feature_list) - set(list(df))
    rootLogger.info("feats to compute:")
    rootLogger.info(feats_to_compute)
    for f in feats_to_compute:
        rootLogger.info(str(f))
        # Feature names are mapped to the functions' names, so it executes only those,
        # that are related to the feats_to_compute!
        df[f] = data.apply(globals()[f], axis=1, raw=True)

    return df


@timeit
def compute_weights_count(data):
    rootLogger.info('computing weights_count')
    data['question1'] = data['question1'].map(lambda x: str(x).lower().split())
    data['question2'] = data['question2'].map(lambda x: str(x).lower().split())

    train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist())

    global words, weights, counts
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}


@timeit
def compute_magic_dict():
    rootLogger.info('magic dict')
    df_train = pd.read_csv('../newdata/train.csv')
    df_train = df_train.fillna(' ')
    df_test = pd.read_csv('../newdata/test.csv')
    ques = pd.concat([df_train[['question1', 'question2']], \
                      df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')
    global q_dict
    for i in range(ques.shape[0]):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    del df_train,df_test


@timeit
def add_magic_feature_freq(df, df_type):
    if 'q1_freq' in set(list(df)):
        return df
    df_train = pd.read_csv('../newdata/train.csv')
    df_train = df_train.fillna(' ')
    df_test = pd.read_csv('../newdata/test.csv')

    global q_dict
    compute_magic_dict()
    def q1_freq(row):
        return (len(q_dict[row['question1']]))

    def q2_freq(row):
        return (len(q_dict[row['question2']]))

    def q1_q2_intersect(row):
        return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    if df_type == 'train':
        df['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
        df['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
        df['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)
    else:
        df['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
        df['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
        df['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

    del df_train, df_test

    return df


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float((len(q1words) + len(q2words)))
    return R


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / float(len(uw)))


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row):
    return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1'])*1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / float(l2):
        return l2 / float(l1)
    else:
        return l1 / float(l2)


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / float(l2):
        return l2 / float(l1)
    else:
        return l1 / float(l2)


def wc_diff_unique_stop(row):
    return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row):
    l1 = len([x for x in set(row['question1']) if x not in stops])*1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / float(l2):
        return l2 / float(l1)
    else:
        return l1 / float(l2)


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / float(l1)
    else:
        return l1 / float(l2)


def char_diff_unique_stop(row):
    return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / float((count + eps))


def tfidf_word_match_share_stops(row):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / float(np.sum(total_weights))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / float(np.sum(total_weights))
    return R


def word_match_share_bigram(row):
    """
    Compute bigram match share.
    :param row: row in the Dataset
    :return: fraction of shared bigrams in both questions.
    """
    q1words = {}
    q2words = {}
    for word in row['bigram1']:
        q1words[word] = 1
    for word in row['bigram2']:
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    try:
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float((len(q1words) + len(q2words)))
    except:
        return 0
    return R


def ner_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['ner1']:
        q1words[word] = 1
    for word in row['ner2']:
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    try:
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float((len(q1words) + len(q2words)))
    except:
        return 0
    return R


def exactly_same(row):
    return int(row['question1'] == row['question2'])


def get_last_word(s):
    return str(s).split()[-1]


def get_len_chrs(s):
    res = -1
    try:
        res = len(set(list(s)))
    except:
        pass
    finally:
        return res


def get_set_chrs(s):
    res = set()
    try:
        res = set(list(s))
    except:
        pass
    finally:
        return res


@timeit
def add_shared_chrs(df, dtype):
    if 'chrs_common_len' in list(df):
        return df

    def get_chrs_common_len(x, raw_df):
        chrs_common = []
        for i, r in raw_df.iterrows():
            try:
                set1 = set(r['question1'])
                set2 = set(r['question2'])
                chrs_common.append(len(set1.intersection(set2)))
            except:
                chrs_common.append(-1)
        x['chrs_common_len'] = chrs_common

    if dtype == 'train':
        df['chrs1_len'] = df.question1.map(lambda x: get_len_chrs(x))
        df['chrs2_len'] = df.question2.map(lambda x: get_len_chrs(x))
        get_chrs_common_len(df, TRAIN)
    else:
        df['chrs1_len'] = df.question1.map(lambda x: get_len_chrs(x))
        df['chrs2_len'] = df.question2.map(lambda x: get_len_chrs(x))
        get_chrs_common_len(df, TEST)

    return df


def edit_distance(row):
    return int(editdistance.eval(row['question1'], row['question2']))


@timeit
def add_svm_feature(df, df_type):
    try:
        df['svm_score'] = pd.read_csv('../newdata/' + df_type + '.pred.csv')['svm_score']
    except:
        pass
        # we cannot compute svm
    return df


@timeit
def compute_ner_feature(df):
    """
    Compute get_ner_feature over the dataset.
    :param df:
    :return:
    """
    import spacy
    nlp = spacy.load('en')

    def get_list_ner(s):
        """
        Computer NER in a question.
        :param s: Question from the Dataset
        :return: list of named entities(only words from the question s).
        """
        res = []
        try:
            doc = nlp(unicode(s))
            for ent in doc.ents:
                #             x = (ent.label_, ent.text)
                res.append(ent.text)
        except:
            pass  # print("err ascii")
        return res

    def get_ner_feature(r):
        """
        Compute NER feature by comparing named entities in the question pair.
        :param r: row in the Dataset
        :return: -1: no NER in both questions
                 -2: NER only in 2nd q.
                 -3: NER only in 1st q.
                 0...1: word_match_share between named entities.
        """
        y = None
        if len(r.ner1) == 0 and len(r.ner2) == 0:
            y = -1.0
        elif len(r.ner1) == 0 and len(r.ner2) != 0:
            y = -2.0
        elif len(r.ner2) == 0 and len(r.ner1) != 0:
            y = -3.0
        else:
            y = ner_word_match_share(r)
        return y

    df['ner1'] = df.question1.map(lambda x: get_list_ner(x))
    print("q1 ner finished")
    df['ner2'] = df.question2.map(lambda x: get_list_ner(x))
    print("q2 ner finished")
    df['ner_wm_share'] = df.apply(lambda r: get_ner_feature(r),axis=1)

    return df


@timeit
def add_ner_feature(df, df_type):
    if 'ner_wm_share' in list(df):
        return df
    try:
        df['ner_wm_share'] = pd.read_csv('../newdata/' + df_type + '.ner_wm.csv')['ner_wm']
    except:
        df = compute_ner_feature(df)
    return df


@timeit
def add_abhishek_features(df,df_type):
    abhishek_feats = pd.read_csv('../newdata/abhishek/' + df_type + '_features.csv')
    x_ab = abhishek_feats.iloc[:, 2:-1]
    x_ab = x_ab.drop('euclidean_distance', axis=1)
    x_ab = x_ab.drop('jaccard_distance', axis=1)
    df = pd.concat((df, x_ab),axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


@timeit
def add_katharina_features(df, df_type):
    katharina_feats = pd.read_csv('../newdata/' + df_type + '_katharina_feats.csv',
                                  converters={'firstthreewords':conv, 'lastwords':conv})
    df['first_three_words'] = katharina_feats['firstthreewords'].astype(float)
    df['last_three_words'] = katharina_feats['lastwords'].astype(float)
    return df


def get_bigrams(q):
    """
    Compute bigram
    :param r: question in the Dataset
    :return: list of tuples of bigrams. E.g. What is the dog? -> [(What, is),(is,the),(the,dog)]
    """
    s = []
    s.append(q)
    res = tuple()
    try:
        res = [b for l in s for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    except:
        pass
    return res


@timeit
def add_bigram_wm_feature(df):
    if 'bigram_wm_share' in list(df):
        return df
    df['bigram1'] = df.question1.map(lambda x: get_bigrams(x))
    df['bigram2'] = df.question2.map(lambda x: get_bigrams(x))
    df['bigram_wm_share'] = df.apply(lambda r: word_match_share_bigram(r), axis=1)
    df.drop('bigram1',axis=1,inplace=True)
    df.drop('bigram2',axis=1,inplace=True)
    return df


def up_down_sampling(X_tr, X_val, y_tr, y_val):
    pos_train = X_tr[y_tr == 1]
    neg_train = X_tr[y_tr == 0]
    X_tr = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))], neg_train))
    y_tr = np.array(
        [0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
    rootLogger.info(np.mean(y_tr))
    del pos_train, neg_train

    pos_valid = X_val[y_val == 1]
    neg_valid = X_val[y_val == 0]
    X_val = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
    y_val = np.array(
        [0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
    rootLogger.info(np.mean(y_valid))
    del pos_valid, neg_valid

    return X_tr, X_val, y_tr, y_val


# read data
rootLogger.info('reading train')
N_train = 10


UP_DOWN_SAMPLING = False

for p_type in PREPROCESSING:
    train = pd.read_csv('../newdata/train.csv', converters={'question1': str, 'question2': str}
                        # ,nrows=N_train
                        )
    train = train.fillna('empty')

    rootLogger.info('Train: ' + str(train.shape))

    # preprocessing
    train_duplicate = train.is_duplicate.values
    rootLogger.info('Preprocessing: ' + str(p_type))
    # setting 1: questions text preprocessing type (stemming, spelling, no preprocessing).
    train = get_preprocessed_data(train, p_type)
    test = pd.read_csv('../newdata/test.csv'
                       # ,nrows=N_train
                       )
    test = test.fillna('empty')
    test = get_preprocessed_data(test, p_type)
    # rootLogger.info('Text sample after preprocessing:')
    # rootLogger.info(train[:10]['question2'])
    # feature engineering
    features_to_compute = ['jaccard',
                           'word_match_share',
                           'common_words',
                           'total_unique_words',
                           'total_unq_words_stop',
                           'wc_diff',
                           'wc_ratio',
                           'wc_diff_unique',
                           'wc_ratio_unique',
                           'wc_diff_unique_stop',
                           'wc_ratio_unique_stop',
                           'same_start_word',
                           'char_diff',
                           'char_ratio',
                           'char_diff_unique_stop',
                           'tfidf_word_match_share_stops',
                           'tfidf_word_match_share',
                           'exactly_same',
                           'edit_distance'
                        ]
    full_feature_list_one = ['first_three_words',
                             'last_three_words',
                             'q1_freq',
                             'q2_freq',
                             'q1_q2_intersect',
                             'chrs_common_len',
                             # 'svm_score',
                             'bigram_wm_share',
                             'ner_wm_share',
                             ]

    full_feature_list_one.extend(features_to_compute)
    rootLogger.info('Feature set to compute:')
    rootLogger.info(full_feature_list_one)

    rootLogger.info("Restoring/computing features...")
    train = get_or_compute_features(train,'train-data_' + p_type + '.csv',features_to_compute, 'train')
    train = add_katharina_features(train,'train') # no p_type
    train = add_magic_feature_freq(train,'train')
    train = add_shared_chrs(train,'train')
    # train = add_svm_feature(train,'train')          # no p_type
    train = add_ner_feature(train,'train')          # no p_type
    train = add_bigram_wm_feature(train)
    train = train[full_feature_list_one]
    train = add_abhishek_features(train,'train')       # no p_type!!! :(
    train.to_csv('../newdata/' + 'train-data_' + p_type + '.csv',index=False)

    rootLogger.info("Full feature set to train on:")
    rootLogger.info(set(list(train)))

    rootLogger.info('Features AUC:')
    for f in set(list(train)):
        try:
            rootLogger.info('AUC ' + str(f) + ': ', roc_auc_score(train_duplicate, train[str(f)].values))
        except Exception as e:
            rootLogger.info('couldn"t compute auc for: ' + str(f))

    # 5-fold CV
    # N_folds = 5
    # kf = KFold(len(train), n_folds=N_folds)
    # for iter_num, (train, test) in enumerate(kf, start=1):
    #     rootLogger.info("%s %s" % (train.shape, test.shape))
    #     rootLogger.info(train)
        #
        # x_train = X_train.ix[train, :]
        # y_train = Y_train[train]
        # x_test = X_train.ix[test, :]
        # y_test = Y_train[test]

        # Rebalancing the Data
        # (x_train, y_train) = oversample_negative(X_train, Y_train)

        # Finally, we split some of the data off for validation
        # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    X_train, X_valid, y_train, y_valid = train_test_split(train, train_duplicate,
                                            test_size=0.2, random_state=RANDOM_STATE)

    if UP_DOWN_SAMPLING:
        X_train, X_valid, y_train, y_valid = up_down_sampling(X_train, X_valid, y_train, y_valid)

    rootLogger.info('Train:' + str(X_train.shape))
    rootLogger.info('Valid: ' + str(X_valid.shape))

    # training
    ROUNDS = 500
    DEPTH = 7
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = DEPTH
    params['silent'] = 1
    params['subsample'] = 0.6
    params['seed'] = RANDOM_STATE

    clr = train_xgb(X_train, X_valid, y_train, y_valid, params,rounds=ROUNDS)
    try:
        clr.save_model('xgb_' + str(ROUNDS) + '_' + str(UP_DOWN_SAMPLING) + '_' + str(DEPTH) + '_.mdl')
    except:
        pass

    preds = predict_xgb(clr, X_valid)
    # evaluation
    rootLogger.info('XGBoost AUC: ' + str(roc_auc_score(y_valid, preds)))
    rootLogger.info('XGBoost logloss: ' + str(log_loss(y_valid,preds)))

    rootLogger.info('Feature importance:')
    rootLogger.info(clr.get_score())

    # prediction for real test set

    # test = pd.read_csv('../newdata/test.csv'
    #                    # ,nrows=N_train
    #                    )
    # test = test.fillna('empty')
    test_id = test['test_id']

    test = get_or_compute_features(test,'test-data_' + p_type + '.csv',features_to_compute,'test')
    test = add_katharina_features(test,'test')
    test = add_magic_feature_freq(test,'test')
    test = add_shared_chrs(test,'test')
    # test = add_svm_feature(test,'test')
    test = add_ner_feature(test,'test')
    # test.to_csv('../newdata/test-data.csv',index=False)
    test = add_bigram_wm_feature(test)
    test = test[full_feature_list_one]
    test = add_abhishek_features(test,'test')
    test.to_csv('../newdata/test-data_' + p_type + '.csv',index=False)
    rootLogger.info('Test features:')
    rootLogger.info(set(list(test)))
    preds = predict_xgb(clr, test)

    timestamp = datetime.now().strftime('-%m-%d-%H:%M')
    sub = pd.DataFrame()
    sub['test_id'] = test_id
    sub['is_duplicate'] = preds
    sub.to_csv('../submissions/sub_' + p_type +  '_' + timestamp + '_.csv',index=False)