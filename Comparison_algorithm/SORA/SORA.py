from GOAndGOA.Ontology import Ontology
import math


class SORA:
    def __init__(self):
        self.bp_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/bp')
        self.cc_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/cc')
        self.mf_onto = Ontology('D:/fang/Achievements/GoGCN/GoGCN_v5/data/ontology/mf')

    def getIC_SORA(self):
        termSet_bp = self.bp_onto.getTermSet()
        descDict_bp = self.bp_onto.getDescendantDict()
        depthDict_bp = self.bp_onto.getDepthDict()

        termSet_cc = self.cc_onto.getTermSet()
        descDict_cc = self.cc_onto.getDescendantDict()
        depthDict_cc = self.cc_onto.getDepthDict()

        termSet_mf = self.mf_onto.getTermSet()
        descDict_mf = self.mf_onto.getDescendantDict()
        depthDict_mf = self.mf_onto.getDepthDict()

        def getIC(termSet, descDict, depthDict):
            ICDict = {}
            total_terms = len(termSet)
            for term in termSet:
                ICDict[term] = 0.0
            for term in termSet:
                depthTerm = max(depthDict[term]) - 1
                descNumTerm = len(descDict[term]) - 1
                ICDict[term] = depthTerm * (1 - math.log(descNumTerm+1)/math.log(total_terms))
            return ICDict

        ICDict_bp = getIC(termSet_bp, descDict_bp, depthDict_bp)
        ICDict_cc = getIC(termSet_cc, descDict_cc, depthDict_cc)
        ICDict_mf = getIC(termSet_mf, descDict_mf, depthDict_mf)

        ICDict = {**ICDict_bp, **ICDict_cc, **ICDict_mf}
        return ICDict


