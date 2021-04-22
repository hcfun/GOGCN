from queue import Queue

class Ontology(object):
    def __init__(self, path):
        """

        Parameter:
        ----------------------
        f: 要处理的原始文件 bp、cc、mf
        """

        self.f = open(path).read().split('[Term]')



    def getTermSet(self):

        """

        Return:
        -------------------------
        TermSet: 术语集合
        """
        termSet = set()
        for split in self.f:
            if 'id: GO:' in split:
                index = split.index('id: GO:')
                termSet.add(split[index+4:index+14])

        return termSet

    def getParentAndSonDict(self):
        """

        Return
        ----------------------------
        parentDict:直接父亲节点
        SonDict:直接儿子节点
        """
        termSet = self.getTermSet()
        parentDict = {}
        sonDict = {}
        for term in termSet:
            parentDict[term] = set()
            sonDict[term] = set()

        for split in self.f:
            if len(split) != 0:
                split = split.split('\n')
                id = split[1][4:]
                for s in split:
                    if 'is_a: GO:' in s:
                        index = s.index('is_a: GO:')
                        if s[index+6: index+16] in termSet:
                            parentDict[id].add(s[index+6: index+16])
                            sonDict[s[index+6: index+16]].add(id)
                    elif 'relationship: part_of GO:' in s:
                        index = s.index('relationship: part_of GO:')
                        if s[index+22: index+32] in termSet:
                            parentDict[id].add(s[index+22: index+32])
                            sonDict[s[index+22: index+32]].add(id)
                    elif 'intersection_of: part_of GO:' in s:
                        index = s.index('intersection_of: part_of GO:')
                        if s[index+25: index+35] in termSet:
                            parentDict[id].add(s[index+25: index+35])
                            sonDict[s[index+25: index+35]].add(id)

        return parentDict, sonDict


    def getRootTerm(self):
        """

        Return
        ------------------------------------
        rootTerm:根节点
        """
        rootTerm = ''
        parDict, _ = self.getParentAndSonDict()
        for term in parDict:
            if len(parDict[term]) == 0:
                rootTerm = term
                break

        return rootTerm

    def getLeafSet(self):
        """

        Return
        --------------------------------
        leafSet:叶子节点term集合
        """
        #叶子节点没有儿子节点
        leafSet = set()
        _, sonDict = self.getParentAndSonDict()
        for term in sonDict:
            if len(sonDict[term]) == 0:
                leafSet.add(term)

        return leafSet


    def getAncestorDict(self):
        """

        Return
        --------------------------------------
        anceDict: 祖先节点
        """
        anceDict = {}
        _, sonDict = self.getParentAndSonDict()
        termSet = self.getTermSet()
        isEnterQueue = {} #入队标志：队内1队外0
        for term in termSet:
            anceDict[term] = set()
            isEnterQueue[term] = 0

        rootTerm = self.getRootTerm()
        queue = Queue()
        front = 0
        rear = 0
        queue.put(rootTerm)
        isEnterQueue[rootTerm] = 1 #根节点入队
        front = 0
        rear =1
        anceDict[rootTerm].add(rootTerm) #将自身加入祖先集合
        while True:
            if front == rear:
                break
            else:
                headTerm = queue.get() #队头出队
                sonOfHeadTermSet = sonDict[headTerm]
                for sonTerm in sonOfHeadTermSet:
                    if isEnterQueue[sonTerm] == 0: #儿子节点不在队里
                        queue.put(sonTerm)
                        isEnterQueue[sonTerm] =1
                        rear += 1
                        anceDict[sonTerm] = anceDict[sonTerm].union(anceDict[headTerm])
                        anceDict[sonTerm].add(sonTerm)
                    elif isEnterQueue[sonTerm] == 1: #儿子节点在队里
                        anceDict[sonTerm] = anceDict[sonTerm].union(anceDict[headTerm])
                        anceDict[sonTerm].add(sonTerm)
                isEnterQueue[headTerm] = 0 #可以再次入队
                front += 1

        return anceDict


    def getDepthDict(self):
        """

        Return
        --------------------------------
        depthDict:深度字典，包含一个term所处的所有深度 key:term value:set()
        """
        depthDict = {}
        _, sonDict = self.getParentAndSonDict()
        termSet = self.getTermSet()
        isEnterQueue = {}  # 入队标志：队内1队外0
        for term in termSet:
            depthDict[term] = set()
            isEnterQueue[term] = 0
        rootTerm = self.getRootTerm()
        queue = Queue()
        front = 0
        rear = 0
        queue.put(rootTerm)
        isEnterQueue[rootTerm] = 1  # 根节点入队
        front = 0
        rear = 1
        depthDict[rootTerm].add(1)
        while True:
            if front == rear:
                break
            else:
                headTerm = queue.get() #队头出队
                sonOfHeadTermSet = sonDict[headTerm]
                for sonTerm in sonOfHeadTermSet:
                    if isEnterQueue[sonTerm] == 0: #儿子节点不在队里
                        queue.put(sonTerm)
                        isEnterQueue[sonTerm] =1
                        rear += 1
                        for depth in depthDict[headTerm]:
                            depthDict[sonTerm].add(depth+1)
                    elif isEnterQueue[sonTerm] == 1:
                        for depth in depthDict[headTerm]:
                            depthDict[sonTerm].add(depth+1)
                isEnterQueue[headTerm] = 0  # 可以再次入队
                front += 1
        return depthDict


    def getDescendantDict(self):
        """

        Return:
        -------------------------------
        descDict:子孙节点  key:term value:set()
        """

        termSet = self.getTermSet()
        leafSet = self.getLeafSet()
        parentDict, _ = self.getParentAndSonDict()
        descDict = {}
        isEnterQueue = {}  # 入队标志：队内1队外0
        for term in termSet:
            descDict[term] = set()
            isEnterQueue[term] = 0
        queue = Queue()
        front = 0
        rear = 0
        #叶子节点入队
        for term in leafSet:
            queue.put(term)
            descDict[term].add(term)
            rear += 1
            isEnterQueue[term] = 1
        while True:
            if front == rear:
                break
            else:
                headTerm = queue.get() #队头出队
                parentOfHeadTermSet = parentDict[headTerm]
                for parentTerm in parentOfHeadTermSet:
                    if isEnterQueue[parentTerm] == 0:#父亲节点不在队里
                        queue.put(parentTerm)
                        isEnterQueue[parentTerm] = 1
                        rear += 1
                        descDict[parentTerm] = descDict[parentTerm].union(descDict[headTerm])
                        descDict[parentTerm].add(parentTerm)
                    elif isEnterQueue[parentTerm] == 1:
                        descDict[parentTerm] = descDict[parentTerm].union(descDict[headTerm])
                        descDict[parentTerm].add(parentTerm)
                isEnterQueue[headTerm] = 0
                front += 1
        return descDict


    # def getLCATerm(self, term1, term2, depthDict, anceDict):
    #     """
    #
    #     :param term1:
    #     :param term2:
    #     :return: LCATerm
    #
    #     1.取出两个节点的祖先节点集合
    #     2.取公共祖先
    #     3.比较公共祖先的最大深度
    #     4.取深度最大的公共祖先
    #     """
    #
    #     # depthDict = self.getDepthDict()
    #     # anceDict = self.getAncestorDict()
    #     unionAnceSet = anceDict[term1] & anceDict[term2]
    #     maxDepth = 0
    #     LCATerm = ''
    #     for term in unionAnceSet:
    #         if max(depthDict[term]) > maxDepth:
    #             maxDepth = max(depthDict[term])
    #             LCATerm = term
    #     return LCATerm















































