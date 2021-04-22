from GOAndGOA.Ontology import Ontology
from queue import Queue





class SV:
    def __init__(self):
        triple_file = open('D:/fang/Achievements/GoGCN/GoGCN_v5/data/triples').read().split('\n')
        self.triples = set()#判断边是is_a还是part_of
        for line in triple_file:
            split = line.split()
            self.triples.add((split[0], split[1], split[2]))
        self.termSet = set()#所有术语（bp,cc,mf）
        term2id_file = open('D:/fang/Achievements/GoGCN/GoGCN_v5/data/term2id').read().split('\n')
        for line in term2id_file:
            split = line.split()
            self.termSet.add(split[0].upper())



    def getSVDict(self):
        """
        计算所有术语节点的sv序列值

        Return
        ----------------------------
        SVDict:term的sv序列值(嵌套字典，外层字典的value也是字典)
        """
        def calSV(rootTerm, sonDict, triples, termSet):
            """
            针对不同的ontology进行SVDict的构造

            Parameter
            ------------------------------------------
            rootTerm:根节点，从上到下操作
            SonDict:儿子字典，记录每一个术语的直接后代
            triples:三元组，为了查询是is_a还是part_of关系
            termSet:术语集合

            Return
            -----------------------------------------
            SVDict:存储sv值的字典
            """

            SVDict = {}
            isEnterQueue = {}

            #为每个term创建字典和入队标志初始化
            for term in termSet:
                SVDict[term] = {}
                isEnterQueue[term] = 0

            queue = Queue()
            front = 0
            rear = 0

            queue.put(rootTerm)
            front = 0
            rear = 1
            isEnterQueue[rootTerm] = 1
            SVDict[rootTerm][rootTerm] = 1.0

            while True:
                if front == rear:
                    break
                else:
                    headTerm = queue.get()#对头出队
                    sonOfHeadTermSet = sonDict[headTerm]#对头术语的儿子
                    for sonTerm in sonOfHeadTermSet:
                        if isEnterQueue[sonTerm] == 0:#儿子术语不在队列中
                            queue.put(sonTerm)
                            isEnterQueue[sonTerm] = 1
                            rear += 1
                            SVDict[sonTerm][sonTerm] = 1.0
                            if (sonTerm, 'is_a', headTerm) in triples:
                                for anceTerm in SVDict[headTerm].keys():
                                    if anceTerm in SVDict[sonTerm].keys():
                                        if SVDict[headTerm][anceTerm]*0.8 > SVDict[sonTerm][anceTerm]:
                                            SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.8
                                    else:
                                        SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.8
                            elif (sonTerm, 'part_of', headTerm) in triples:
                                for anceTerm in SVDict[headTerm].keys():
                                    if anceTerm in SVDict[sonTerm].keys():
                                        if SVDict[headTerm][anceTerm]*0.6 > SVDict[sonTerm][anceTerm]:
                                            SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.6
                                    else:
                                        SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.6

                        elif isEnterQueue[sonTerm] == 1:#儿子术语在队列中
                            if (sonTerm, 'is_a', headTerm) in triples:
                                for anceTerm in SVDict[headTerm].keys():
                                    if anceTerm in SVDict[sonTerm].keys():
                                        if SVDict[headTerm][anceTerm]*0.8 > SVDict[sonTerm][anceTerm]:
                                            SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.8
                                    else:
                                        SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.8
                            elif (sonTerm, 'part_of', headTerm) in triples:
                                for anceTerm in SVDict[headTerm].keys():
                                    if anceTerm in SVDict[sonTerm].keys():
                                        if SVDict[headTerm][anceTerm]*0.6 > SVDict[sonTerm][anceTerm]:
                                            SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm]*0.6
                                    else:
                                        SVDict[sonTerm][anceTerm] = SVDict[headTerm][anceTerm] * 0.6

                    isEnterQueue[headTerm] = 0
                    front += 1

            return SVDict

        bp_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/bp')
        cc_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/cc')
        mf_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/mf')

        bpRootTerm = bp_onto.getRootTerm()
        ccRootTerm = cc_onto.getRootTerm()
        mfRootTerm = mf_onto.getRootTerm()

        bpParDict, bpSonDict = bp_onto.getParentAndSonDict()
        ccParDict, ccSonDict = cc_onto.getParentAndSonDict()
        mfParDict, mfSonDict = mf_onto.getParentAndSonDict()

        bpTermSet = bp_onto.getTermSet()
        ccTermSet = cc_onto.getTermSet()
        mfTermSet = mf_onto.getTermSet()

        bpSVDict = calSV(rootTerm=bpRootTerm, sonDict=bpSonDict, triples=self.triples, termSet=bpTermSet)
        ccSVDict = calSV(rootTerm=ccRootTerm, sonDict=ccSonDict, triples=self.triples, termSet=ccTermSet)
        mfSVDict = calSV(rootTerm=mfRootTerm, sonDict=mfSonDict, triples=self.triples, termSet=mfTermSet)

        SVDict = {**bpSVDict, **ccSVDict, **mfSVDict}
        return SVDict





