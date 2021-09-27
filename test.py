import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import numpy as np
import codecs
import time
import torch
import argparse


from TransE_pytoch import dataloader,entities2id,relations2id


def test_data_loader(entity_embedding_file, relation_embedding_file, test_data_file):
    print("load data...")
    file1 = entity_embedding_file
    file2 = relation_embedding_file
    file3 = test_data_file

    entity_dic = {}
    relation_dic = {}
    triple_list = []

    with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity_dic[int(line[0])] = json.loads(line[1])


        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation_dic[int(line[0])] = json.loads(line[1])

    with codecs.open(file3, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            head = int(entities2id[triple[0]])

            relation = int(relations2id[triple[1]])
            tail = int(entities2id[triple[2]])

            triple_list.append([head, relation, tail])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_dic.keys()), len(relation_dic.keys()), len(triple_list)))

    return entity_dic, relation_dic, triple_list

class testSeTransE:
    def __init__(self, entities_dict, relations_dict, test_triple_list, train_triple_list, valid_triple, filter_triple=False, norm=2):
        self.entities = entities_dict
        self.relations = relations_dict
        self.test_triples = test_triple_list
        self.train_triples = train_triple_list
        self.valid_triples = valid_triple
        self.filter = filter_triple
        self.norm = norm

        self.mean_rank = 0
        self.mean_reciprocal_rank = 0
        self.hit_10 = 0
        self.hit_1 = 0
        self.hit_3 = 0

    def test_run(self):

        print(self.relations)
        # hits = 0
        # rank_sum = 0
        # num = 0
        # for triple in self.test_triples:
        #     start = time.time()
        #     num += 1
        #     print(num, triple)
        #     rank_head_dict = {}
        #     rank_tail_dict = {}
        #
        #     if self.filter:
        #         head_filter = []
        #         tail_filter = []
        #         for tr in self.train_triples:
        #             if tr[1] == triple[1] and tr[2] == triple[2]:
        #                 head_filter.append(tr)
        #             if tr[0] == triple[0] and tr[1] == triple[1]:
        #                 tail_filter.append(tr)
        #         for tr in self.test_triples:
        #             if tr[1] == triple[1] and tr[2] == triple[2]:
        #                 head_filter.append(tr)
        #             if tr[0] == triple[0] and tr[1] == triple[1]:
        #                 tail_filter.append(tr)
        #
        #     #
        #     for entity in self.entities.keys():
        #         head_triple = [entity, triple[1], triple[2]]
        #         if self.filter:
        #             if head_triple in head_filter:
        #                 continue
        #         head_embedding = self.entities[head_triple[0]]
        #         tail_embedding = self.entities[head_triple[2]]
        #         relation_embedding = self.relations[head_triple[1]]
        #         distance = self.distance(head_embedding, relation_embedding, tail_embedding)
        #         rank_head_dict[tuple(head_triple)] = distance
        #
        #     for tail in self.entities.keys():
        #         tail_triple = [triple[0], triple[1], tail]
        #         if self.filter:
        #             if tail_triple in tail_filter:
        #                 continue
        #         head_embedding = self.entities[tail_triple[0]]
        #         relation_embedding = self.relations[tail_triple[1]]
        #         tail_embedding = self.entities[tail_triple[2]]
        #         distance = self.distance(head_embedding, relation_embedding, tail_embedding)
        #         rank_tail_dict[tuple(tail_triple)] = distance
        hits = 0
        hits_1 = 0
        hits_3 = 0
        reciprocal_sum = 0
        rank_sum = 0
        num = 0

        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            head_embedding = []
            tail_embedding = []
            relation_embedding = []
            tamp = []

            head_filter = []
            tail_filter = []
            if self.filter:

                for tr in self.train_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.test_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.valid_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)


            for i, entity in enumerate(self.entities.keys()):

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(self.entities[head_triple[0]])
                tail_embedding.append(self.entities[head_triple[2]])
                relation_embedding.append(self.relations[head_triple[1]])
                tamp.append(tuple(head_triple))

            distance = self.distance(head_embedding, relation_embedding, tail_embedding)

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            relation_embedding = []
            tamp = []

            for i, tail in enumerate(self.entities.keys()):

                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(self.entities[tail_triple[0]])
                relation_embedding.append(self.relations[tail_triple[1]])
                tail_embedding.append(self.entities[tail_triple[2]])
                tamp.append(tuple(tail_triple))

            distance = self.distance(head_embedding, relation_embedding, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]

            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''

            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            # calculate the mean_rank and hit_10
            # head data
            i = 0
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 1:
                        hits_1 += 1
                    if i < 3:
                        hits_3 += 1
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    reciprocal_sum = reciprocal_sum + 1 + 1/(i + 1)
                    break

            # tail rank
            i = 0
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0][2]:
                    if i < 1:
                        hits_1 += 1
                    if i < 3:
                        hits_3 += 1
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    reciprocal_sum = reciprocal_sum + 1 + 1/(i + 1)
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_1 = hits_1 / (2 * len(self.test_triples))
        self.hit_3 = hits_3 / (2 * len(self.test_triples))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))
        self.mean_reciprocal_rank = reciprocal_sum
        return self.hit_10, self.hit_1, self.hit_3, self.mean_reciprocal_rank, self.mean_rank


    # def distance(self, h, r, t):
    #     head = np.array(h)
    #     relation = np.array(r)
    #     tail = np.array(t)
    #     d = head + relation - tail
    #     if self.norm == 1:
    #         return np.sum(np.fabs(d))
    #         # return np.linalg.norm(d, ord=1)
    #     else:
    #         return np.sum(np.square(d))
    #         return np.linalg.norm(d, ord=2)

    def distance(self, h, r, t):
        head = torch.from_numpy(np.array(h))
        rel = torch.from_numpy(np.array(r))
        tail = torch.from_numpy(np.array(t))
        #print("head=", head)
        #distance = head + rel - tail
        distance_t_h = tail - head
        score_t_h = torch.norm(distance_t_h, p=self.norm, dim=1)
        score_r = torch.norm(rel, p=self.norm, dim=1)
        distance = np.fabs(score_t_h-score_r)
        #score = torch.norm(distance, p=self.norm, dim=1)
        #print("score=: ", score.numpy)
        #return score.numpy()
        print("distance=", distance)
        return  distance

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Testing Knowledge Graph Embedding Models',
        usage='SeTransE.py [<args>] [-h | --help]'
    )
    parser.add_argument('--do_test', action='store_true', default='do_test')
    parser.add_argument('--dataset', type=str, default='WN18')
    return parser.parse_args(args)

def main(args):

    if (not args.do_test):
        # raise ValueError('one of train/val/test mode must be choosed.')
        print("no choosed")
    if args.do_test:
        print('do_test')

        if args.dataset == 'WN18':
            file1 = "WN18\\wordnet-mlj12-train.txt"
            file2 = "WN18\\entity2id.txt"
            file3 = "WN18\\relation2id.txt"
            file4 = "WN18\\wordnet-mlj12-valid.txt"
            entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4, args.dataset)
            entity_dict, relation_dict, test_triple = test_data_loader("SeTransE_entity_final",
                                                                       "SeTransE_relation_final", "WN18\\wordnet-mlj12-test.txt")
            test = testSeTransE(entity_dict, relation_dict, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
            hit10, hit1, hit3, mrr, mean_rank = test.test_run()
            print("raw entity hits@1: ", hit1)
            print("raw entity hits@3: ", hit3)
            print("raw entity hits@10: ", hit10)
            print("raw entity mrr: ", mrr)
            print("raw entity meanrank: ",mean_rank)
        if args.dataset == 'WN18rr':
            file1 = "wn18rr\\train.txt"
            file2 = "wn18rr\\entities.dict"
            file3 = "wn18rr\\relations.dict"
            file4 = "wn18rr\\valid.txt"
            entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4, args.dataset)
            entity_dict, relation_dict, test_triple = test_data_loader("SeTransE_entity_final",
                                                                       "SeTransE_relation_final", "wn18rr\\test.txt")
            test = testSeTransE(entity_dict, relation_dict, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
            hit10, hit1, hit3, mrr, mean_rank = test.test_run()
            print("raw entity hits@1: ", hit1)
            print("raw entity hits@3: ", hit3)
            print("raw entity hits@10: ", hit10)
            print("raw entity mrr: ", mrr)
            print("raw entity meanrank: ",mean_rank)
        if args.dataset == 'YAGO3-10':
            file1 = "YAGO3-10\\train.txt"
            file2 = "YAGO3-10\\entities.dict"
            file3 = "YAGO3-10\\relations.dict"
            file4 = "YAGO3-10\\valid.txt"
            entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4, args.dataset)
            entity_dict, relation_dict, test_triple = test_data_loader("SeTransE_entity_final",
                                                                       "SeTransE_relation_final", "YAGO3-10\\test.txt")
            test = testSeTransE(entity_dict, relation_dict, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
            hit10, hit1, hit3, mrr, mean_rank = test.test_run()
            print("raw entity hits@1: ", hit1)
            print("raw entity hits@3: ", hit3)
            print("raw entity hits@10: ", hit10)
            print("raw entity mrr: ", mrr)
            print("raw entity meanrank: ", mean_rank)
        if args.dataset == 'FB15k':
            file1 = "FB15k\\freebase_mtr100_mte100-train.txt"
            file2 = "FB15k\\entity2id.txt"
            file3 = "FB15k\\relation2id.txt"
            file4 = "FB15k\\freebase_mtr100_mte100-valid.txt"
            entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4, args.dataset)
            entity_dict, relation_dict, test_triple = test_data_loader("SeTransE_entity_final",
                                                                       "SeTransE_relation_final", "FB15k\\freebase_mtr100_mte100-test.txt")
            test = testSeTransE(entity_dict, relation_dict, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
            hit10, hit1, hit3, mrr, mean_rank = test.test_run()
            print("raw entity hits@1: ", hit1)
            print("raw entity hits@3: ", hit3)
            print("raw entity hits@10: ", hit10)
            print("raw entity mrr: ", mrr)
            print("raw entity meanrank: ", mean_rank)
        if args.dataset == 'FB15k-237':
            file1 = "FB15k-237\\train.txt"
            file2 = "FB15k-237\\entities.dict"
            file3 = "FB15k-237\\relations.dict"
            file4 = "FB15k-237\\valid.txt"
            entity_set, relation_set, train_triple, valid_triple = dataloader(file1, file2, file3, file4, args.dataset)
            entity_dict, relation_dict, test_triple = test_data_loader("SeTransE_entity_final",
                                                                       "SeTransE_relation_final", "FB15k-237\\test.txt")
            test = testSeTransE(entity_dict, relation_dict, test_triple, train_triple, valid_triple, filter_triple=False, norm=2)
            hit10, hit1, hit3, mrr, mean_rank = test.test_run()
            print("raw entity hits@1: ", hit1)
            print("raw entity hits@3: ", hit3)
            print("raw entity hits@10: ", hit10)
            print("raw entity mrr: ", mrr)
            print("raw entity meanrank: ", mean_rank)

    return 0


if __name__ == "__main__":

    main(parse_args())

