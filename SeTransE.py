import codecs
import numpy as np
import copy
import time
import random
import argparse

entities2id = {}
relations2id = {}


def dataloader(file1, file2, file3, dataset):
    print("load file...")

    entity = []
    relation = []
    with open(file2, 'r', encoding='utf-8') as f1, open(file3, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            if dataset == "FB15k":
                entities2id[line[0]] = line[1]
                entity.append(line[1])
            if dataset == "WN18":
                entities2id[line[0]] = line[1]
                entity.append(line[1])
            if dataset == "FB15k-237":
                entities2id[line[1]] = line[0]
                entity.append(line[0])
            if dataset == "WN18rr":
                entities2id[line[1]] = line[0]
                entity.append(line[0])
            if dataset == "YAGO3-10":
                entities2id[line[1]] = line[0]
                entity.append(line[0])
        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            if dataset == "FB15k":
                relations2id[line[0]] = line[1]
                relation.append(line[1])
            if dataset == "WN18":
                relations2id[line[0]] = line[1]
                relation.append(line[1])
            if dataset == "FB15k-237":
                relations2id[line[1]] = line[0]
                relation.append(line[0])
            if dataset == "WN18rr":
                relations2id[line[1]] = line[0]
                relation.append(line[0])
            if dataset == "YAGO3-10":
                relations2id[line[1]] = line[0]
                relation.append(line[0])

    triple_list = []
    with codecs.open(file1, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            if dataset == "FB15k":
                h_ = entities2id[triple[0]]
                r_ = relations2id[triple[1]]
                t_ = entities2id[triple[2]]
                triple_list.append([h_, r_, t_])

            if dataset == "WN18":
                h_ = entities2id[triple[0]]
                r_ = relations2id[triple[1]]
                t_ = entities2id[triple[2]]
                triple_list.append([h_, r_, t_])

            if dataset == "WN18rr":
                h_ = entities2id[triple[0]]
                r_ = relations2id[triple[1]]
                t_ = entities2id[triple[2]]
                triple_list.append([h_, r_, t_])

            if dataset == "FB15k-237":
                h_ = entities2id[triple[0]]
                r_ = relations2id[triple[1]]
                t_ = entities2id[triple[2]]
                triple_list.append([h_, r_, t_])

            if dataset == "YAGO3-10":
                h_ = entities2id[triple[0]]
                r_ = relations2id[triple[1]]
                t_ = entities2id[triple[2]]
                triple_list.append([h_, r_, t_])




    print("Complete load. entity : %d , relation : %d , triple : %d" % (
    len(entity), len(relation), len(triple_list)))

    return entity, relation, triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    #return np.sum(np.square(h + r - t))
    dist_t_h = np.sqrt(np.sum(np.square(t - h)))
    dist_r = np.sqrt(np.sum(np.square(r)))

    #print(dist_t_h-dist_r)
    return  np.fabs(dist_t_h-dist_r)

class SeTransE:
    def __init__(self, entity, relation, triple_list, ep=500, tbs=100, embedding_dim=50, lr=0.01, margin=1.0, norm=1):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.epochs = ep
        self.nbatches = tbs
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0


    def data_initialise(self):
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            entityVectorList[entity] = entity_vector

        for relation in self.relations:
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                                self.dimension)
            relation_vector = self.normalization(relation_vector)
            relationVectorList[relation] = relation_vector

        self.entities = entityVectorList
        self.relations = relationVectorList

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    def training_run(self, out_file_title = ''):

        batch_size = int(len(self.triples) / self.nbatches)
        print("batch size: ", batch_size)
        for epoch in range(self.epochs):
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for entity in self.entities.keys():
                self.entities[entity] = self.normalization(self.entities[entity])

            for batch in range(self.nbatches):
                batch_samples = random.sample(self.triples, batch_size)

                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]

                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        with codecs.open("SeTransE_entity_final", "w") as f1:

            for e in self.entities.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entities[e])))
                f1.write("\n")

        with codecs.open("SeTransE_relation_final", "w") as f2:
            for r in self.relations.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relations[r])))
                f2.write("\n")

    def update_triple_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的elements都不同
        copy_entity = copy.deepcopy(self.entities)
        copy_relation = copy.deepcopy(self.relations)

        for correct_sample, corrupted_sample in Tbatch:

            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[2]]
            relation_copy = copy_relation[correct_sample[1]]

            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[2]]

            correct_head = self.entities[correct_sample[0]]
            correct_tail = self.entities[correct_sample[2]]
            relation = self.relations[correct_sample[1]]

            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[2]]

            # calculate the distance of the triples
            if self.norm == 1:
                correct_distance = norm_l1(correct_head, relation, correct_tail)
                corrupted_distance = norm_l1(corrupted_head, relation, corrupted_tail)

            else:

                #relation_corrected = (correct_tail - correct_head) * np.sqrt(np.sum(np.square(relation)))/np.sqrt(np.sum(np.square(correct_tail-correct_head)))
                correct_distance = norm_l2(correct_head, relation, correct_tail)
                #relation_corrupted = (corrupted_tail - corrupted_head) * np.sqrt(np.sum(np.square(relation)))/np.sqrt(np.sum(np.square(corrupted_tail-corrupted_head)))
                corrupted_distance = norm_l2(corrupted_head, relation, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            #loss = self.margin + np.fabs(correct_distance) - np.fabs(corrupted_distance)

            if loss > 0:
                self.loss += loss

                #lr_r = 1 * np.fabs(np.fabs(correct_distance)-np.sqrt(np.sum(np.square(relation))))/np.sqrt(np.sum(np.square(relation)))
                #lr_r_c =  1 * np.fabs(np.fabs(corrupted_distance)-np.sqrt(np.sum(np.square(relation))))/np.sqrt(np.sum(np.square(relation)))
                # rotate r

                relation = (correct_tail - correct_head) * np.sqrt(np.sum(np.square(relation)))/np.sqrt(np.sum(np.square(correct_tail-correct_head)))

                correct_gradient = 2 * (correct_head + relation - correct_tail)
                # rotate r
                relation = (corrupted_tail - corrupted_head) * np.sqrt(np.sum(np.square(relation)))/np.sqrt(np.sum(np.square(corrupted_tail-corrupted_head)))
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)
                #correct_gradient = 2 * (correct_head - correct_tail )*lr_r
                #corrupted_gradient = 2 * (corrupted_head - corrupted_tail)*lr_r_c

                if self.norm == 1:
                    for i in range(len(correct_gradient)):
                        if correct_gradient[i] > 0:
                            correct_gradient[i] = 1
                        else:
                            correct_gradient[i] = -1

                        if corrupted_gradient[i] > 0:
                            corrupted_gradient[i] = 1
                        else:
                            corrupted_gradient[i] = -1


                relation_copy = (correct_tail - correct_head) * np.sqrt(np.sum(np.square(relation)))/np.sqrt(np.sum(np.square(correct_tail-correct_head)))
                correct_copy_head -= self.learning_rate * correct_gradient
                relation_copy -= self.learning_rate * correct_gradient
                correct_copy_tail -= -1 * self.learning_rate * correct_gradient

                relation_copy = (corrupted_tail - corrupted_head) * np.sqrt(np.sum(np.square(relation_copy)))/np.sqrt(np.sum(np.square(corrupted_tail-corrupted_head)))
                relation_copy -= -1 * self.learning_rate * corrupted_gradient
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replaces the tail entity, the head entity's embedding need to be updated twice
                    correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replaces the head entity, the tail entity's embedding need to be updated twice
                    corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    correct_copy_tail -= self.learning_rate * corrupted_gradient

                # normalising these new embedding vector, instead of normalising all the embedding together
                copy_entity[correct_sample[0]] = self.normalization(correct_copy_head)
                copy_entity[correct_sample[2]] = self.normalization(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity's embedding
                    copy_entity[corrupted_sample[2]] = self.normalization(corrupted_copy_tail)
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replace the head entity, update the head entity's embedding
                    copy_entity[corrupted_sample[0]] = self.normalization(corrupted_copy_head)
                # the paper mention that the relation's embedding don't need to be normalised
                copy_relation[correct_sample[1]] = relation_copy
                # copy_relation[correct_sample[1]] = self.normalization(relation_copy)

        self.entities = copy_entity
        self.relations = copy_relation

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='SeTransE.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true', default='do_train')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--dataset', type=str, default='FB15k')
    parser.add_argument('-tbs', default=100, type=int)
    parser.add_argument('-dim', default=50, type=int)
    parser.add_argument('-mg', default=1.0, type=float)
    parser.add_argument('-lr', default=0.00005, type=float)
    parser.add_argument('-ep', default=500, type=int)
    return parser.parse_args(args)

def main(args):

    if (not args.do_train)  and (not args.do_test):
        # raise ValueError('one of train/val/test mode must be choosed.')
        print("no choosed")
    if args.do_train:
        print('do_train')
        if args.dataset == 'WN18':
            file1 = "WN18\\wordnet-mlj12-train.txt"
            file2 = "WN18\\entity2id.txt"
            file3 = "WN18\\relation2id.txt"
            entity_set, relation_set, triple_list = dataloader(file1, file2, file3, args.dataset)

            transE = SeTransE(entity_set, relation_set, triple_list, args.ep, args.tbs, args.dim, args.lr, args.mg, norm=2, )
            transE.data_initialise()
            transE.training_run(out_file_title="")
        if args.dataset == 'FB15k':
            file1 = "FB15k\\freebase_mtr100_mte100-train.txt"
            file2 = "FB15k\\entity2id.txt"
            file3 = "FB15k\\relation2id.txt"
            entity_set, relation_set, triple_list = dataloader(file1, file2, file3, args.dataset)

            transE = SeTransE(entity_set, relation_set, triple_list, args.ep, args.tbs, args.dim, args.lr, args.mg, norm=2, )
            transE.data_initialise()
            transE.training_run(out_file_title="")
        if args.dataset == 'FB15k-237':
            file1 = "FB15k-237\\train.txt"
            file2 = "FB15k-237\\entities.dict"
            file3 = "FB15k-237\\relations.dict"
            entity_set, relation_set, triple_list = dataloader(file1, file2, file3, args.dataset)

            transE = SeTransE(entity_set, relation_set, triple_list, args.ep, args.tbs, args.dim, args.lr, args.mg, norm=2, )
            transE.data_initialise()
            transE.training_run(out_file_title="")
        if args.dataset == 'WN18rr':
            file1 = "wn18rr\\train.txt"
            file2 = "wn18rr\\entities.dict"
            file3 = "wn18rr\\relations.dict"
            entity_set, relation_set, triple_list = dataloader(file1, file2, file3, args.dataset)

            transE = SeTransE(entity_set, relation_set, triple_list, args.ep, args.tbs, args.dim, args.lr, args.mg, norm=2, )
            transE.data_initialise()
            transE.training_run(out_file_title="")
        if args.dataset == 'YAGO3-10':
            file1 = "YAGO3-10\\train.txt"
            file2 = "YAGO3-10\\entities.dict"
            file3 = "YAGO3-10\\relations.dict"
            entity_set, relation_set, triple_list = dataloader(file1, file2, file3, args.dataset)

            transE = SeTransE(entity_set, relation_set, triple_list, args.ep, args.tbs, args.dim, args.lr, args.mg, norm=2, )
            transE.data_initialise()
            transE.training_run(out_file_title="")


    return 0

if __name__ == '__main__':

    main(parse_args())



    # file1 = "WN18\\wordnet-mlj12-train.txt"
    # file2 = "WN18\\entity2id.txt"
    # file3 = "WN18\\relation2id.txt"




