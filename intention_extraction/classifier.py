import nltk
import sklearn
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.svm import LinearSVC

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


class Classifier:

    def __init__(self, final_test):
        self.final_test = final_test

    def loading_data(self, file_path):
        with open(file_path) as file:
            full_list = []
            label_list = []
            raw_list = []
            pos_list = []
            parameter_list = []
            arg_list = []

            for line in file:
                if line.startswith('# ::id'):
                    temp_list = [line]
                elif line == '\n' and temp_list != []:
                    append_bool = True
                    for item in temp_list:
                        if '/ and' in item:
                            append_bool = False
                    for item in temp_list:
                        if ':intent' in item and append_bool:
                            label_list.append(
                                item.replace(' ', '').replace(')', '').replace(':intent“', '').replace('”\n', ''))
                    if append_bool:
                        full_list.append(tuple(temp_list))
                        raw_list.append(temp_list[1].replace('# ::snt ', '').replace('\n', ''))
                        parameter_list.append(
                            [p for p in temp_list[2].replace('# ::parameters ', '').replace('\n', '').split(' ')])
                        arg_list.append([arg for arg in temp_list if arg.startswith('    :ARG')])
                    temp_list = []
                else:
                    temp_list.append(line)

        for sent in raw_list:
            tok_sent = nltk.word_tokenize(sent)
            temp_list = nltk.pos_tag(tok_sent)
            pos_list.append([t for (w, t) in [s for s in temp_list]])

        return full_list, label_list, raw_list, pos_list, parameter_list, arg_list

    def get_features(self, raw_sent, pos, arg):
        fw = raw_sent.split(' ')[0]
        arg_word_list = []
        for item in arg:
            arg_word = re.sub(r'[^/]+/', '', item)
            arg_word_list.append(re.sub(r'[^a-z]', '', arg_word))

        features = {
            'first_word': fw,
            'first_pos': pos[0],
            'first_is_verb': pos[0] == 'VB',
            'Arg_count': len(arg),
            '1st_arg_word': arg_word_list[0] if len(arg_word_list) > 0 else 'NONE',
            '2nd_arg_word': arg_word_list[1] if len(arg_word_list) > 1 else 'NONE',
            '3rd_arg_word': arg_word_list[2] if len(arg_word_list) > 2 else 'NONE'
        }
        print(features)
        return features

    def generate_tdt(self, label_list, raw_list, pos_list, parameter_list, arg_list, x, y, z):
        train_sents = list(range(x))
        dev_sents = list(range(y))
        test_sents = list(range(z))
        for i in range(x):
            train_sents[i] = (raw_list[i], pos_list[i], label_list[i], parameter_list[i], arg_list[i])
        for j in range(y):
            dev_sents[j] = (raw_list[j + x], pos_list[j + x], label_list[j + x], parameter_list[j + x], arg_list[j + x])
        for k in range(z):
            test_sents[k] = (
                raw_list[k + x + y], pos_list[k + x + y], label_list[k + x + y], parameter_list[k + x + y],
                arg_list[k + x + y])
        return train_sents, dev_sents, test_sents

    def intent_clf(self):
        vec = DictVectorizer()

        X_train = vec.fit_transform([self.get_features(r, p, a) for (r, p, l, ps, a) in train_sents])
        y_train = [l for (r, p, l, ps, a) in train_sents]

        X_dev = vec.transform([self.get_features(r, p, a) for (r, p, l, ps, a) in dev_sents])
        y_dev = [l for (r, p, l, ps, a) in dev_sents]

        print("Start training...")
        clf = OneVsRestClassifier(LinearSVC(multi_class='ovr')).fit(X_train, y_train)
        y_pred = clf.predict(X_dev)
        labels = list(clf.classes_)
        print("Labels for test:", labels)

        print("Acc score on dev set: {0:.3g}".format(accuracy_score(y_dev, y_pred)))
        print("Recall score on dev set: {0:.3g}".format(recall_score(y_dev, y_pred, average='macro')))
        print("F1 score on dev set: {0:.3g}".format(f1_score(y_dev, y_pred, average='macro')))

        if self.final_test:
            X_test = vec.transform([self.get_features(r, p, a) for (r, p, l, ps, a) in test_sents])
            y_test = [l for (r, p, l, ps, a) in test_sents]
            y_pred = clf.predict(X_test)

            print("Acc score on test set: {0:.3g}".format(accuracy_score(y_test, y_pred)))
            print("Recall score on test set: {0:.3g}".format(recall_score(y_test, y_pred, average='macro')))
            print("F1 score on test set: {0:.3g}".format(f1_score(y_test, y_pred, average='macro')))

    def word2features(self, sent, i):
        word = sent[0].split(' ')[i]
        pos = sent[1][i]
        intent = sent[2]
        is_arg = False
        for a in sent[4]:
            if word in a:
                is_arg = True

        features = {
            'bias': 1.0,
            'word': word,
            'pos': pos,
            'intent': intent,
            'is_arg': is_arg
        }
        if i > 0:
            word1 = sent[0].split(' ')[i - 1]
            pos1 = sent[1][i - 1]
            features.update({
                '-1:word': word1,
                '-1:pos': pos1
            })
        else:
            features['BOS'] = True

        if i < len(sent[1]) - 1:
            word1 = sent[0].split(' ')[i + 1]
            pos1 = sent[1][i + 1]
            features.update({
                '+1:word': word1,
                '+1:pos': pos1
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent[1]))]

    def sent2labels(self, sent):
        p_list = [ps for (r, p, l, ps, a) in sent]
        raw_list = [r for (r, p, l, ps, a) in sent]
        return_me = []
        for s in raw_list:
            return_me.append(['0'] * len(s.split(' ')))
        for i, ps in enumerate(p_list):
            for p in ps:
                p = p.replace('x', '')
                if p != '0' and p != 'ap0':
                    p = int(p)
                    return_me[i][p - 1] = '1'
        return return_me

    def parameter_clf(self):
        X_train = [self.sent2features(s) for s in train_sents]
        y_train = self.sent2labels(train_sents)
        print(X_train)
        print(y_train)

        X_dev = [self.sent2features(s) for s in dev_sents]
        y_dev = self.sent2labels(dev_sents)

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        crf.fit(X_train, y_train)
        labels = list(crf.classes_)
        y_pred = crf.predict(X_dev)
        print("F1 for dev set: ", metrics.flat_f1_score(y_dev, y_pred, average='weighted', labels=labels))

        if self.final_test:
            X_test = [self.sent2features(s) for s in test_sents]
            y_test = self.sent2labels(test_sents)

            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=True
            )
            params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
            }
            f1_scorer = make_scorer(metrics.flat_f1_score,
                                    average='weighted', labels=labels)
            rs = RandomizedSearchCV(crf, params_space,
                                    cv=3,
                                    verbose=1,
                                    n_jobs=-1,
                                    n_iter=50,
                                    scoring=f1_scorer)
            rs.fit(X_train, y_train)
            crf = rs.best_estimator_
            y_pred = crf.predict(X_test)
            print(metrics.flat_classification_report(
                y_test, y_pred, labels=labels, digits=3
            ))


if __name__ == '__main__':
    # True for test result, False for dev result only
    classifier = Classifier(True)
    full_list, label_list, raw_list, pos_list, parameter_list, arg_list = classifier.loading_data(
        'huric302_intent_shuffled')
    train_sents, dev_sents, test_sents = classifier.generate_tdt(label_list, raw_list, pos_list, parameter_list,
                                                                 arg_list, 206, 30, 30)
    classifier.intent_clf()
    classifier.parameter_clf()
