"""
Calculate the gene similarity of gene pairs
"""
from GOAndGOA.Annotation import Annotation
from GOAndGOA.Ontology import Ontology
from Comparison_algorithm.Wang.SV import SV
from Comparison_algorithm.SORA.SORA import SORA
from helper import *

class geneSim:
    def __init__(self, params):
        self.p = params
        self.term_embed = np.loadtxt((self.p.embed_dir), delimiter=',', skiprows=0)  # 术语的分布式表示
        self.genePair_file = open(self.p.genePair).read().split('\n')  # 基因对
        self.term2id = {}  # 术语和id对应，方便提取相应的分布式表示
        for line in open('./data/term2id').read().split('\n'):
            split = line.split()
            self.term2id[split[0].upper()] = split[1]
        self.bp_onto = Ontology('./data/ontology/bp')
        self.cc_onto = Ontology('./data/ontology/cc')
        self.mf_onto = Ontology('./data/ontology/mf')
        self.anceDict_bp = self.bp_onto.getAncestorDict()
        self.descDict_bp = self.bp_onto.getDescendantDict()
        self.depthDict_bp = self.bp_onto.getDepthDict()
        self.anceDict_cc = self.cc_onto.getAncestorDict()
        self.descDict_cc = self.cc_onto.getDescendantDict()
        self.depthDict_cc = self.cc_onto.getDepthDict()
        self.anceDict_mf = self.mf_onto.getAncestorDict()
        self.descDict_mf = self.mf_onto.getDescendantDict()
        self.depthDict_mf = self.mf_onto.getDepthDict()

        self.goa = Annotation(self.p.goa_dir)
        # 取出基因注释 BP CC MF 避免多次读取
        self.bpIEADict, self.bpNIEADict, self.ccIEADict, self.ccNIEADict, self.mfIEADict, self.mfNIEADict = self.goa.getAnnotationDict_Direct()
        self.bpIEADict_Ance, self.bpNIEADict_Ance, self.ccIEADict_Ance, self.ccNIEADict_Ance, self.mfIEADict_Ance, self.mfNIEADict_Ance = self.goa.getAnnotationDict_Ance()

        self.SVDict = SV().getSVDict()#Wang的SV字典
        self.ICDict_SORA = SORA().getIC_SORA()#SORA的ic字典




    def calSimTerms_GOGCN(self, t1, t2):
        t1_embed = self.term_embed[int(self.term2id[t1])]
        t2_embed = self.term_embed[int(self.term2id[t2])]
        numerator = np.sum(t1_embed * t2_embed)
        denominator1 = np.sum(t1_embed.__pow__(2)).__pow__(0.5)
        denominator2 = np.sum(t2_embed.__pow__(2)).__pow__(0.5)
        sim = numerator/(denominator1*denominator2)


        return sim

    def calSimTerms_Wang(self, term1, term2):

        unionAnceTermSet = self.SVDict[term1].keys() & self.SVDict[term2].keys()
        numerator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        for term in unionAnceTermSet:
            numerator += (self.SVDict[term1][term] + self.SVDict[term2][term])
        for v in self.SVDict[term1].values():
            denominator1 += v
        for v in self.SVDict[term2].values():
            denominator2 += v
        sim = numerator / (denominator1 + denominator2)

        return sim

    def calSimTerms_Resnik(self, term1, term2, depthDict, anceDict):
        unionAnceSet = anceDict[term1] & anceDict[term2]
        maxDepth = 0
        LCATerm = ''
        for term in unionAnceSet:
            if max(depthDict[term]) > maxDepth:
                maxDepth = max(depthDict[term])
                LCATerm = term
        sim = self.ICDict_SORA[LCATerm]

        return sim


#计算基因之间的相似度

    def calSimGenes_Resnik(self, gene1, gene2):
        def calSimGenes(set1, set2, depthDict, anceDict):
            if len(set1) == 0 or len(set2) == 0:
                sim = 0.0
            else:
                sumsim = 0.0
                for term1 in set1:
                    for term2 in set2:
                        sumsim += self.calSimTerms_Resnik(term1, term2, depthDict, anceDict)
                sim = sumsim / (len(set1) * len(set2))
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict_Ance[gene1]
            annotSet2_bpiea = self.bpIEADict_Ance[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea, self.depthDict_bp, self.anceDict_bp)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict_Ance[gene1]
            annotSet2_bpniea = self.bpNIEADict_Ance[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea, self.depthDict_bp, self.anceDict_bp)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict_Ance[gene1]
            annotSet2_cciea = self.ccIEADict_Ance[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea, self.depthDict_cc, self.anceDict_cc)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict_Ance[gene1]
            annotSet2_ccniea = self.ccNIEADict_Ance[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea, self.depthDict_cc, self.anceDict_cc)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict_Ance[gene1]
            annotSet2_mfiea = self.mfIEADict_Ance[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea, self.depthDict_mf, self.anceDict_mf)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict_Ance[gene1]
            annotSet2_mfniea = self.mfNIEADict_Ance[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea, self.depthDict_mf, self.anceDict_mf)

            return sim_mfiea, sim_mfniea


    def calSimGenes_SimGIC(self, gene1, gene2):
        def calSimGenes(set1, set2):
            interTermSet = set1 & set2
            unionTermSet = set1 | set2
            if len(interTermSet) == 0 or len(unionTermSet) == 0:
                sim = 0.0
            else:
                numerator = 0.0
                denominator = 0.0
                for term in interTermSet:
                    numerator += self.ICDict_SORA[term]
                for term in unionTermSet:
                    denominator += self.ICDict_SORA[term]
                if denominator == 0:
                    sim = 0.0
                else:
                    sim = numerator/denominator
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict_Ance[gene1]
            annotSet2_bpiea = self.bpIEADict_Ance[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict_Ance[gene1]
            annotSet2_bpniea = self.bpNIEADict_Ance[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict_Ance[gene1]
            annotSet2_cciea = self.ccIEADict_Ance[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict_Ance[gene1]
            annotSet2_ccniea = self.ccNIEADict_Ance[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict_Ance[gene1]
            annotSet2_mfiea = self.mfIEADict_Ance[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict_Ance[gene1]
            annotSet2_mfniea = self.mfNIEADict_Ance[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea)

            return sim_mfiea, sim_mfniea




    def calSimGenes_SORA(self, gene1, gene2):

        def getCetTermSet(set, descDict):
            """
            Return
            ---------------------------
            cetTermSet：cet术语集合
            """
            cetTermSet = set
            for term in set:
                for descTerm in descDict[term]:
                    i = 0
                    if i == 2:
                        cetTermSet.remove(term)
                        break
                    else:
                        if descTerm in set:
                            i += 1
            return cetTermSet

        def calICSet(Set, descDict, anceDict):
            #取出CET集合
            cetTermSet = getCetTermSet(Set, descDict)
            unionTermSet = set()
            IC_Set = 0.0
            for term in cetTermSet:
                if len(unionTermSet) == 0:
                    IC_Set += self.ICDict_SORA[term]
                    unionTermSet = unionTermSet & anceDict[term]
                else:
                    interSet = unionTermSet | anceDict[term]
                    lcaSet = getCetTermSet(interSet, descDict)
                    lcaTerm = lcaSet.pop()
                    IC_Set += (self.ICDict_SORA[term] - self.ICDict_SORA[lcaTerm])
                    unionTermSet = unionTermSet.union(anceDict[term])
            return IC_Set

        def calSimGenes(set1, set2, descDict, anceDict):
            #取出两个集合的交集
            interTermSet = set1 & set2
            if len(set1) == 0 or len(set2) == 0 or len(interTermSet) == 0:
                sim = 0.0
            else:
                IC_inter = calICSet(interTermSet, descDict, anceDict)
                IC_set1 = calICSet(set1, descDict, anceDict)
                IC_set2 = calICSet(set2, descDict, anceDict)
                if IC_set1 == 0 and IC_set2 != 0:
                    sim = (IC_inter / IC_set2) / 2
                elif IC_set2 == 0 and IC_set1 != 0:
                    sim = (IC_inter / IC_set1) / 2
                elif IC_set1 == 0 and IC_set2 == 0:
                    sim = 0
                else:
                    sim = (IC_inter/IC_set1 + IC_inter/IC_set2)/2
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict_Ance[gene1]
            annotSet2_bpiea = self.bpIEADict_Ance[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea, self.descDict_bp, self.anceDict_bp)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict_Ance[gene1]
            annotSet2_bpniea = self.bpNIEADict_Ance[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea, self.descDict_bp, self.anceDict_bp)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict_Ance[gene1]
            annotSet2_cciea = self.ccIEADict_Ance[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea, self.descDict_cc, self.anceDict_cc)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict_Ance[gene1]
            annotSet2_ccniea = self.ccNIEADict_Ance[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea, self.descDict_cc, self.anceDict_cc)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict_Ance[gene1]
            annotSet2_mfiea = self.mfIEADict_Ance[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea, self.descDict_mf, self.anceDict_mf)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict_Ance[gene1]
            annotSet2_mfniea = self.mfNIEADict_Ance[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea, self.descDict_mf, self.anceDict_mf)

            return sim_mfiea, sim_mfniea
        


    def calSimGenes_VSM(self, gene1, gene2):
        def calSimGenes(set1, set2):
            if set1 == set2 and len(set1) != 0 and len(set2) != 0:
                sim = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                sim = 0.0
            else:
                v1 = np.zeros((len(self.term2id)), dtype=int)
                v2 = np.zeros((len(self.term2id)), dtype=int)
                for term in set1:
                    v1[int(self.term2id[term])] = 1
                for term in set2:
                    v2[int(self.term2id[term])] = 1
                numerator = np.sum(v1 * v2)
                denominator1 = np.sum(v1.__pow__(2)).__pow__(0.5)
                denominator2 = np.sum(v2.__pow__(2)).__pow__(0.5)
                sim = numerator / (denominator1 * denominator2)
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict_Ance[gene1]
            annotSet2_bpiea = self.bpIEADict_Ance[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict_Ance[gene1]
            annotSet2_bpniea = self.bpNIEADict_Ance[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict_Ance[gene1]
            annotSet2_cciea = self.ccIEADict_Ance[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict_Ance[gene1]
            annotSet2_ccniea = self.ccNIEADict_Ance[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict_Ance[gene1]
            annotSet2_mfiea = self.mfIEADict_Ance[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict_Ance[gene1]
            annotSet2_mfniea = self.mfNIEADict_Ance[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea)

            return sim_mfiea, sim_mfniea




    def calSimGenes_SimUI(self, gene1, gene2):
        def calSimGenes(set1, set2):
            if set1 == set2 and len(set1) != 0 and len(set2) != 0:
                sim = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                sim = 0.0
            else:
                intersectionSet = set1 & set2#交集
                unionSet = set1 | set2#并集
                sim = len(intersectionSet)/len(unionSet)

            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict_Ance[gene1]
            annotSet2_bpiea = self.bpIEADict_Ance[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict_Ance[gene1]
            annotSet2_bpniea = self.bpNIEADict_Ance[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict_Ance[gene1]
            annotSet2_cciea = self.ccIEADict_Ance[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict_Ance[gene1]
            annotSet2_ccniea = self.ccNIEADict_Ance[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict_Ance[gene1]
            annotSet2_mfiea = self.mfIEADict_Ance[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict_Ance[gene1]
            annotSet2_mfniea = self.mfNIEADict_Ance[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea)

            return sim_mfiea, sim_mfniea






    def calSimGenes_Wang(self, gene1, gene2):

        #计算基因之间相似度
        def calSimGenes(set1, set2):
            """
            计算两个基因的相似度（由两个术语集合注释）

            Return:
            -----------------------------------
            基因相似度
            """
            if set1 == set2 and len(set1) != 0 and len(set2) != 0:
                sim = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                sim = 0.0
            else:
                simTermDict = {}
                for term1 in set1:
                    for term2 in set2:
                        if term1 == term2:
                            simTermDict[(term1, term2)] = 1.0
                        else:
                            simTermDict[(term1, term2)] = self.calSimTerms_Wang(term1, term2)

                simg1g2 = 0.0
                for term1 in set1:
                    simt1g2 = 0.0
                    for term2 in set2:
                        simt1t2 = simTermDict[(term1, term2)]
                        if simt1t2 > simt1g2:
                            simt1g2 = simt1t2
                    simg1g2 += simt1g2

                simg2g1 = 0.0
                for term2 in set2:
                    simt2g1 = 0.0
                    for term1 in set1:
                        simt2t1 = simTermDict[(term1, term2)]
                        if simt2t1 > simt2g1:
                            simt2g1 = simt2t1
                    simg2g1 += simt2g1
                sim = (simg1g2 + simg2g1) / (len(set1) + len(set2))
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict[gene1]
            annotSet2_bpiea = self.bpIEADict[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict[gene1]
            annotSet2_bpniea = self.bpNIEADict[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict[gene1]
            annotSet2_cciea = self.ccIEADict[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict[gene1]
            annotSet2_ccniea = self.ccNIEADict[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict[gene1]
            annotSet2_mfiea = self.mfIEADict[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict[gene1]
            annotSet2_mfniea = self.mfNIEADict[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea)

            return sim_mfiea, sim_mfniea

    def calSimGenes_GOGCN(self, gene1, gene2):

        def calSimGenes(set1, set2):
            """
            计算两个基因的相似度（由两个术语集合注释）

            Return:
            -----------------------------------
            基因相似度
            """
            if set1 == set2 and len(set1) != 0 and len(set2) != 0:
                sim = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                sim = 0.0
            else:

                simTermDict = {}
                for term1 in set1:
                    for term2 in set2:
                        if term1 == term2:
                            simTermDict[(term1, term2)] = 1.0
                        else:
                            simTermDict[(term1, term2)] = self.calSimTerms_GOGCN(term1, term2)

                simg1g2 = 0.0
                for term1 in set1:
                    simt1g2 = 0.0
                    for term2 in set2:
                        simt1t2 = simTermDict[(term1, term2)]
                        if simt1t2 > simt1g2:
                            simt1g2 = simt1t2
                    simg1g2 += simt1g2

                simg2g1 = 0.0
                for term2 in set2:
                    simt2g1 = 0.0
                    for term1 in set1:
                        simt2t1 = simTermDict[(term1, term2)]
                        if simt2t1 > simt2g1:
                            simt2g1 = simt2t1
                    simg2g1 += simt2g1
                sim = (simg1g2 + simg2g1) / (len(set1) + len(set2))
            return sim

        if self.p.onto == 'bp':
            # bp_iea
            annotSet1_bpiea = self.bpIEADict[gene1]
            annotSet2_bpiea = self.bpIEADict[gene2]
            sim_bpiea = calSimGenes(annotSet1_bpiea, annotSet2_bpiea)

            # bp_niea
            annotSet1_bpniea = self.bpNIEADict[gene1]
            annotSet2_bpniea = self.bpNIEADict[gene2]
            sim_bpniea = calSimGenes(annotSet1_bpniea, annotSet2_bpniea)

            return sim_bpiea, sim_bpniea

        elif self.p.onto == 'cc':
            # cc_iea
            annotSet1_cciea = self.ccIEADict[gene1]
            annotSet2_cciea = self.ccIEADict[gene2]
            sim_cciea = calSimGenes(annotSet1_cciea, annotSet2_cciea)

            # cc_niea
            annotSet1_ccniea = self.ccNIEADict[gene1]
            annotSet2_ccniea = self.ccNIEADict[gene2]
            sim_ccniea = calSimGenes(annotSet1_ccniea, annotSet2_ccniea)

            return sim_cciea, sim_ccniea

        elif self.p.onto == 'mf':
            # mf_iea
            annotSet1_mfiea = self.mfIEADict[gene1]
            annotSet2_mfiea = self.mfIEADict[gene2]
            sim_mfiea = calSimGenes(annotSet1_mfiea, annotSet2_mfiea)

            # mf_niea
            annotSet1_mfniea = self.mfNIEADict[gene1]
            annotSet2_mfniea = self.mfNIEADict[gene2]
            sim_mfniea = calSimGenes(annotSet1_mfniea, annotSet2_mfniea)

            return sim_mfiea, sim_mfniea

    def calSimGenes_Allmethods(self):
        gene1List = []  # 第一列基因
        gene2List = []  # 第二列基因
        for line in self.genePair_file:
            split = line.split()
            gene1List.append(split[0])
            gene2List.append(split[1])
        #计算相似度
        iea = open('./geneSim/{}/{}_iea_{}'.format(self.p.sim_dictory, self.p.onto, self.p.genePair[-3:]), 'w')
        niea = open('./geneSim/{}/{}_niea_{}'.format(self.p.sim_dictory, self.p.onto, self.p.genePair[-3:]), 'w')
        iea.write('Gene1' + '\t' + 'Gene2' + '\t' + 'GOGCN' + '\t' + 'Wang' + '\t' + 'simUI' + '\t' + 'VSM' + '\t' + 'SORA' + '\t'+ 'simGIC' + '\t' + 'Resnik' +'\n')
        niea.write('Gene1' + '\t' + 'Gene2' + '\t' + 'GOGCN' + '\t' + 'Wang' + '\t' + 'simUI' + '\t' + 'VSM' + '\t' + 'SORA' + '\t'+ 'simGIC' + '\t' + 'Resnik' +'\n')
        for i in range(len(gene1List)):
            gene1 = gene1List[i]
            gene2 = gene2List[i]
            sim_iea, sim_niea = self.calSimGenes_GOGCN(gene1, gene2)
            sim_iea_wang, sim_niea_wang = self.calSimGenes_Wang(gene1, gene2)
            sim_iea_ui, sim_niea_ui = self.calSimGenes_SimUI(gene1, gene2)
            sim_iea_vsm, sim_niea_vsm = self.calSimGenes_VSM(gene1, gene2)
            sim_iea_sora, sim_niea_sora = self.calSimGenes_SORA(gene1, gene2)
            sim_iea_gic, sim_niea_gic = self.calSimGenes_SimGIC(gene1, gene2)
            sim_iea_resnik, sim_niea_resnik = self.calSimGenes_Resnik(gene1,gene2)

            iea.write(gene1 + '\t' + gene2 + '\t' + '{}'.format(sim_iea) + '\t' + '{}'.format(sim_iea_wang) + '\t' + '{}'.format(sim_iea_ui) + '\t' + '{}'.format(sim_iea_vsm) + '\t' + '{}'.format(sim_iea_sora) + '\t' + '{}'.format(sim_iea_gic) +  '\t' + '{}'.format(sim_iea_resnik) + '\n')
            niea.write(gene1 + '\t' + gene2 + '\t' + '{}'.format(sim_niea) + '\t' + '{}'.format(sim_niea_wang) + '\t' + '{}'.format(sim_niea_ui) + '\t' + '{}'.format(sim_niea_vsm) + '\t' + '{}'.format(sim_niea_sora) + '\t' + '{}'.format(sim_niea_gic)+  '\t' + '{}'.format(sim_niea_resnik) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-genePair', dest='genePair', default='./data/genePair/genePair_sgd_pathway', help='genePair')
    parser.add_argument('-ontology', dest='onto', default='mf', help='bp,cc,mf')
    parser.add_argument('-sim_dictory', dest='sim_dictory', default='pathway_sgd', help='dictory of storing the experimental results')
    parser.add_argument('-embed_dir', dest='embed_dir', default='./embedding/ent_embedding_Pretraining_75b2d53a.csv', help='embeddings of terms')
    parser.add_argument('-goa_dir', dest='goa_dir', default='./data/goa_sgd', help='goa_file:goa_sgd for sgd, goa_human for human')
    args = parser.parse_args()

    genesim = geneSim(args)
    genesim.calSimGenes_Allmethods()