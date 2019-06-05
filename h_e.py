import glob
import nltk
from nltk.corpus import treebank
from nltk.corpus import dependency_treebank
import random


def filter(tree):
    for subtree in tree.subtrees():
        if subtree.label() == 'S-IMP':
            return True
            # else:
            #     print(subtree.label())
    return False


def check_imp():
    filter_sents = []
    for tree in treebank.parsed_sents():
        if filter(tree):
            filter_sents.append(tree)
    print(filter_sents)


def get_raw_sents():
    return_me = []
    for filename in glob.glob('gathered/*.xml'):
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == 2:
                    token_list = line.replace('<sentence>', '').replace('</sentence>\n', '')
                    return_me.append(token_list)
    return return_me


gold_dependency_list = []


def get_gold_dependency():
    for filename in glob.glob('gathered/*.xml'):
        gd_list = []
        with open(filename) as f:
            marker = False
            for line in f:
                if line.replace('\n', '') == '<dependencies>':
                    marker = True
                elif line.replace('\n', '') == '</dependencies>':
                    marker = False
                elif marker:
                    gd_list.append(line.split(' ')[3].replace('type="', '').replace('"/>\n', ''))
        gold_dependency_list.append(gd_list)


dependency_list = []


def get_dependency():
    with open('input.txt.tok.charniak.parse.dep') as f:
        d_list = []
        for line in f:
            if line != '\n':
                d_list.append(line.split('(')[0])
            else:
                dependency_list.append(d_list)
                d_list = []


def difference_check(gold, test):
    return_me = 0
    length = min(len(gold), len(test))
    for i in range(length):
        if gold[i] != test[i]:
            return_me += 1
    return_me += abs(len(gold) - len(test))
    return return_me


def dependency_func():
    raw_sents = get_raw_sents()
    get_gold_dependency()
    print(gold_dependency_list)
    get_dependency()
    print(dependency_list)

    count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 0: 0}
    acc_list = []
    for i, gold in enumerate(gold_dependency_list):
        diff = difference_check(gold, dependency_list[i])
        count_dict[diff] += 1
        acc = (len(gold) - diff) / len(gold)
        acc_list.append(acc)
        if diff > 2:
            print(raw_sents[i])
            print(gold, dependency_list[i])
    print(len(gold_dependency_list))
    print(count_dict)
    print(sum(acc_list) / float(len(acc_list)))

    for i, list in enumerate(gold_dependency_list):
        if 'dojb' in list:
            print("locate ", i)

    for i, list in enumerate(dependency_list):
        if 'amod' in list:
            print(i, dependency_list[i], raw_sents[i])


def input_fix():
    output_list = []
    with open('huric.input', 'r') as f:
        for line in f:
            rp_dict = {"can you ": "",
                       "could you ": "",
                       "would you ": "",
                       "please ": "",
                       "please": "",
                       "thank you": "",
                       "thanks": "",
                       "for me": "",
                       "john ": "",
                       "hey ": "",
                       "robot ": "",
                       "michael ": "",
                       "  ": " "}
            temp = line
            for i, j in rp_dict.items():
                temp = temp.replace(i, j)
            output_list.append(temp)
    with open('huric_fix', 'w') as f:
        for line in output_list:
            f.write(line)


def random_shuffle(file_name):
    label_list = ['Check_Action', 'Define_Action', 'Follow_Action', 'Grab_Action', 'New_State_Action',
                  'Relocate_Action', 'Search_Action']
    with open(file_name) as file:
        full_list = []
        for line in file:
            if line.startswith('# ::id'):
                temp_list = [line]
            elif line == '\n' and temp_list != []:
                full_list.append(tuple(temp_list))
                temp_list = []
            else:
                temp_list.append(line)

    found_label1 = []
    found_label2 = []

    while len(found_label1) < 7 or len(found_label2) < 7 :
        print(len(found_label1))
        print(len(found_label2))
        found_label1 = set([])
        found_label2 = set([])
        random.shuffle(full_list)
        for c, item in enumerate(full_list):
            if c < 75:
                for label in label_list:
                    if label in item[4]:
                        found_label1.add(label)
            if 75 < c < 150:
                for label in label_list:
                    if label in item[4]:
                        found_label2.add(label)
    return full_list


def write_to_file(file_name, list, train_size, dev_size, test_size):
    with open(file_name + '_train', 'w') as file:
        for count, item in enumerate(list):
            if count < train_size:
                for line in item:
                    file.write(line)
                file.write('\n')

    with open(file_name + '_dev', 'w') as file:
        for count, item in enumerate(list):
            if train_size <= count < train_size + dev_size:
                for line in item:
                    file.write(line)
                file.write('\n')

    with open(file_name + '_test', 'w') as file:
        for count, item in enumerate(list):
            if train_size + dev_size <= count < train_size + dev_size + test_size:
                for line in item:
                    file.write(line)
                file.write('\n')


# input_fix()
write_to_file('huric302_intent_shuffled', random_shuffle('huric302_intent.txt'), 302, 0, 0)
