# -*- coding: utf-8 -*-
import argparse, csv, regex, re, sys
from collections import defaultdict
import random
import numpy as np
from itertools import izip
import os.path
import sqlite3
from nltk.tokenize import sent_tokenize, word_tokenize

def remove_punctuation(text):
    return regex.sub(ur"\p{P}+", " ", text)

FTP = ' FTP '
BUZZ = 'bbuuzz'
ftp = ["for 10 points", "for ten points", 'ftp', 'for 15 points', '10 points']
def clean_text(q):
    # remove pronunciation guides and other formatting extras
    q = q.replace(' (*) ', ' ')
    q = q.replace('\n ', ' ')
    q = q.replace('\\n ', ' ')
    q = q.replace('mt. ', 'mt ')
    q = q.replace('{', '')
    q = q.replace('}', '')
    q = q.replace('~', '')
    q = q.replace('(*)', '')
    q = q.replace('*', '')
    # utf8 punct to ascii
    q = q.replace(u'“', '"')
    q = q.replace(u'”', '"')
    q = q.replace(u'‘', '"')
    q = q.replace(u'’', '"')
    q = q.replace(u'，', ',')
    # sentence delimeter in qb2
    #q = q.replace(' ||| ', ' ')
    # don't remove BUZZ position
    q = re.sub(r'\[[^\]]*%s[^\]]*\]' % BUZZ, BUZZ, q)
    q = re.sub(r'\[[^\]]*\]', '', q)
    q = re.sub(r'\([^\)]*%s[^\)]*\)' % BUZZ, BUZZ, q)
    q = re.sub(r'\([^\)]*\)', '', q)
    q = re.sub(r'\?+', '', q)

    ss = [' '.join(word_tokenize(s)) for s in sent_tokenize(q)]
    q = ' '.join(ss)

    q = re.sub(r'[ ]+', ' ', q).strip().lower()

    for phrase in ftp:
        q = q.replace(phrase, FTP)
    return q

class BuzzError(Exception):
    pass

def map_buzz_pos(buzz_pos, q):
    '''
    map buzz pos relative to original text to cleaned text
    '''
    words = q.split(' ')
    # buzz_pos starts from 1
    try:
        words[buzz_pos-1] = '%s %s' % (words[buzz_pos-1], BUZZ)
    except IndexError:
        print 'q:', q
        print 'buzz pos:', buzz_pos
        print 'word:', len(words)
        raise BuzzError
    #    return 0
    q = clean_text(' '.join(words))
    # get new buzz position
    ss = q.split()
    old_buzz_pos = buzz_pos
    # buzz_pos starts from 1
    buzz_pos = None
    for i, s in enumerate(ss):
        if s == BUZZ:
            buzz_pos = i
            break
    if buzz_pos is None:
        print 'original:', old_buzz_pos, words
        print 'cleaned:', q
        raise BuzzError
    assert buzz_pos is not None
    return buzz_pos

def assign_fold(probs, n):
    # probs = p(train), p(dev), p(test)
    # n = number of examples
    ntrain = int(probs[0] * n)
    # no test data
    if probs[2] == 0:
        ntest = 0
        if n < 2:
            ntrain = 1
            ndev = 0
        else:
            ndev = n - ntrain
            if ndev == 0:
                ntrain -= 1
                ndev += 1
            assert ntrain > 0 and ndev > 0
    else:
        if n < 3:
            ntrain = 1
            ntest = n - ntrain
            ndev = 0
        else:
            ndev = max(int(probs[1] * n), 1)
            ntest = n - ntrain - ndev
            if ntest == 0:
                ntrain -= 1
                ntest += 1
            assert ntrain > 0 and ndev > 0 and ntest > 0
    folds = ['train']*ntrain + ['dev']*ndev + ['test']*ntest
    random.shuffle(folds)
    return folds

def load_buzzes(buzz_file, user_cutoff):
    user_buzzes = defaultdict(dict)
    with open(buzz_file, 'r') as fin:
        # header
        fields = {v: k for k, v in enumerate(fin.readline().strip().split(','))}
        qid_field = fields['Question ID']
        uid_field = fields['User ID']
        pos_field = fields['Buzz Position']
        correct_field = fields['Correct']
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            qid = int(row[qid_field])
            uid = row[uid_field]
            position = int(float(row[pos_field]))
            correct = int(row[correct_field])
            # remove duplicated questions answered by the same user
            # use the one with a later buzz
            if qid not in user_buzzes[uid] or \
                    position > user_buzzes[uid][qid][0]:
                user_buzzes[uid][qid] = (position, correct)
    print 'load buzzes from', buzz_file
    print '#user,#q', len(user_buzzes), len(set([x for qlist in user_buzzes.values() for x in qlist]))

    # filter buzzes
    for uid, qdict in user_buzzes.items():
        if len(qdict) < user_cutoff:
            del user_buzzes[uid]
    buzzes = defaultdict(list)
    for uid, qdict in user_buzzes.items():
        for qid, buzz in qdict.items():
            position, correct = buzz
            buzzes[qid].append([uid, position, correct])
    print 'after removing users answered fewer than %d questions:' % user_cutoff
    print '#user,#q,#buzz/q', len(user_buzzes), len(buzzes), sum([len(x) for x in buzzes.values()]) / float(len(buzzes))
    return buzzes

def get_col_names(c, table_name):
    cols = {}
    for row in c.execute('PRAGMA table_info(%s)' % table_name):
        cols[row[1]] = row[0]
    return cols

def filter_answer(questions, ans_question, ans_cutoff):
    for ans, q in ans_question.items():
        if len(q) < ans_cutoff:
            for qid in q:
                if qid in questions:
                    del questions[qid]
            del ans_question[ans]
    print 'after removing answers with fewer than %d questions:' % args.ans_cutoff
    num_questions = sum([len(x) for x in ans_question.values()])
    assert(len(questions) == num_questions)
    print '#ans,#q:', len(ans_question.keys()), len(questions)
    return questions, ans_question

def load_questions_db(question_db, ans_cutoff):
    questions = {}
    ans_question = defaultdict(list)

    assert os.path.isfile(question_db)
    conn = sqlite3.connect(question_db)
    c_q = conn.cursor()
    c_t = conn.cursor()
    questions_cols = get_col_names(c_q, 'questions')
    text_cols = get_col_names(c_t, 'text')

    for row_q in c_q.execute('select * from questions where page != ""'):
        ans = row_q[questions_cols['page']]
        # remove arithmetic questions
        if '(number)' in ans:
            continue
        qid = row_q[questions_cols['id']]
        category = row_q[questions_cols['category']]
        sents = []
        for row_t in c_t.execute('select * from text where question == %d' % qid):
            sents.append(row_t[text_cols['raw']])
        text = ' '.join(sents)
        # remove too short questions
        if qid in questions or len(text.split()) < 15:
            continue
        questions[qid] = [ans, text, category]
        ans_question[ans].append(qid)
    print 'load quesitons (db) from', question_db
    print '#ans,#q:', len(ans_question.keys()), len(questions)

    return filter_answer(questions, ans_question, ans_cutoff)

def load_questions(qfile, ans_cutoff):
    questions = {}
    ans_question = defaultdict(list)

    with open(qfile, 'r') as fin:
        # header
        fields = {v: k for k, v in enumerate(fin.readline().strip().split(','))}
        qid_field = fields['Question ID']
        ans_field = fields['Wikipedia Page']
        text_field = fields['Text']
        cat_field = fields['Category']
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            id_ = int(row[qid_field])
            ans = row[ans_field]
            text = row[text_field]
            text = text.replace(' ||| ', ' ')
            cat = row[cat_field]
            # remove too short questions
            if id_ in questions or len(text.split()) < 10:
                continue
            questions[id_] = [ans, text, cat]
            ans_question[ans].append(id_)
    print 'load quesitons from', qfile
    print '#ans,#q:', len(ans_question.keys()), len(questions)

    return filter_answer(questions, ans_question, ans_cutoff)

def split(ans_qids, probs, have_buzz=False):
    fold_count = defaultdict(int)
    users = defaultdict(set)
    num_buzzes = 0
    sum_buzz_pos = 0
    for ans, q in ans_qids.items():
        folds = assign_fold(probs, len(q))
        for qid, fold in izip(q, folds):
            questions[qid].append(fold)
            fold_count[fold] += 1
            qtext = questions[qid][1].strip()
            if have_buzz:
                # update buzz position relative to cleaned text
                for i, buzz in enumerate(buzzes[qid]):
                    uid, buzz_pos, correct = buzz
                    users[fold].add(uid)
                    try:
                        new_buzz_pos = map_buzz_pos(buzz_pos, qtext)
                    except BuzzError:
                        print qid, uid
                        sys.exit()
                    sum_buzz_pos += new_buzz_pos
                    num_buzzes += 1
                    buzz[1] = new_buzz_pos
                    assert(buzzes[qid][i][1] == new_buzz_pos)
            questions[qid][1] = clean_text(qtext)
            qlen = len(questions[qid][1].split())
            for i, buzz in enumerate(buzzes[qid]):
                assert buzz[1] <= qlen

    if have_buzz:
        print 'mean buzz position:', float(sum_buzz_pos) / num_buzzes

    # stats for train, dev, test
    str_format = '{:<10}{:<10}{:<10}{:<10}'
    print str_format.format('split', '#example', '#user', '#new')
    for fold, count in fold_count.items():
        if have_buzz:
            nusers = len(users[fold])
            nnew = nusers - len(users[fold].intersection(users['train']))
        else:
            nusers = '-'
            nnew = '-'
        print str_format.format(fold, count, nusers, nnew)

def write_example(qids, output_file, have_buzz=False):
    # sort questions by length (for minimum padding)
    qids = sorted(qids, key=lambda i: len(questions[i][1].split()))
    with open(output_file, 'w') as fout:
        for qid in qids:
            ans, qtext, cat, fold = questions[qid]
            if have_buzz:
                buzz = '|'.join(['-'.join([str(x) for x in buzz]) for buzz in buzzes[qid]])
                fout.write(('%s\n' % (' ||| '.join([str(qid), cat, ans, fold, qtext, buzz]))).encode('utf8'))
            else:
                fout.write(('%s\n' % (' ||| '.join([str(qid), cat, ans, fold, qtext]))).encode('utf8'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buzz', help='buzzes.csv')
    parser.add_argument('--question', help='question file')
    parser.add_argument('--users', help='file of user ids')
    parser.add_argument('--ans_cutoff', type=int, default=10, help='remove answers who have fewer than 10 questions')
    parser.add_argument('--ans_filter', help='a file of pre-selected answers: only keep questions with answers in this set')
    parser.add_argument('--user_cutoff', type=int, default=10, help='remove users who have answered fewer than 10 questions')
    parser.add_argument('--content_train_frac', type=float, default=0.8, help='training fraction')
    parser.add_argument('--content_dev_frac', type=float, default=0.1, help='validation fraction')
    parser.add_argument('--buzz_train_frac', type=float, default=0.8, help='training fraction')
    parser.add_argument('--buzz_dev_frac', type=float, default=0.1, help='validation fraction')
    parser.add_argument('--output_dir', help='path of output file')
    args = parser.parse_args()
    random.seed(100)

    # load questions
    if args.question.endswith('.db'):
        questions, ans_question = load_questions_db(args.question, args.ans_cutoff)
    else:
        questions, ans_question = load_questions(args.question, args.ans_cutoff)

    # load buzzes
    buzzes = load_buzzes(args.buzz, args.user_cutoff)

    # filter ans based on buzz data
    buzz_ans = set()
    num_no_question = 0
    for qid in buzzes:
        if qid in questions:
            buzz_ans.add(questions[qid][0])
        else:
            num_no_question += 1
    print '%d answers with buzzes' % len(buzz_ans)
    print '%d questions with buzzes does not have training data' % num_no_question

    # collect content model and buzz model data
    content_ans_qids = defaultdict(list)
    buzz_ans_qids = defaultdict(list)
    for ans in buzz_ans:
        for qid in ans_question[ans]:
            if qid in buzzes:
                buzz_ans_qids[ans].append(qid)
            else:
                content_ans_qids[ans].append(qid)
        # remove answers without additional question
        if ans not in content_ans_qids:
            del buzz_ans_qids[ans]
    print 'content model #ans,#q', len(content_ans_qids), sum([len(x) for x in content_ans_qids.values()])
    print 'buzz model #ans,#q', len(buzz_ans_qids), sum([len(x) for x in buzz_ans_qids.values()])

    # process question text and assign folds
    buzz_probs = [args.buzz_train_frac, args.buzz_dev_frac, 1.0-args.buzz_dev_frac-args.buzz_train_frac]
    split(buzz_ans_qids, buzz_probs, True)
    content_probs = [args.content_train_frac, args.content_dev_frac, 1.0-args.content_dev_frac-args.content_train_frac]
    split(content_ans_qids, content_probs, False)

    # print
    content_qids = [x for qlist in content_ans_qids.values() for x in qlist]
    buzz_qids = [x for qlist in buzz_ans_qids.values() for x in qlist]
    write_example(content_qids, args.output_dir+'/content_data.txt', False)
    write_example(buzz_qids, args.output_dir+'/buzz_data.txt', True)
