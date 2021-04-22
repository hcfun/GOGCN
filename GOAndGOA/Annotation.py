from GOAndGOA.Ontology import Ontology


class Annotation:
    def __init__(self, path):
        """

        Parameter:
        ------------------------------------------
        path:要处理的文件的路径：goa_human、goa_cere
        """
        self.f = open(path).read().split('\n')


    def getGeneSet(self):
        """

        Return:
        --------------------------------------------
        geneSet:基因集合
        """
        geneSet = set()
        for line in self.f:
            split = line.split('\t')
            geneSet.add(split[2])
        return geneSet

    def getProtein2GeneDict(self):
        """

        Return:
        ------------------------------------------------
        pro2GeneDict:蛋白质和基因对应的字典


        """
        pro2geneDict = {}
        for line in self.f:
            split = line.split('\t')
            pro2geneDict[split[1]] = split[2]
        return pro2geneDict



    def getAnnotationDict_Direct(self):
        """
        不包含祖先的注释字典，pairwise方法
        Return:
        ----------------------------------------------------
        bpIEADict: IEA注释的bp
        bpNIEADict: NIEA注释的bp
        ccIEADict:IEA注释的cc
        ccNIEADict:NIEA注释的cc
        mfIEADict:IEA注释的mf
        mfNIEADict:NIEA注释的mf
        """
        geneSet = self.getGeneSet()
        bpIEADict = {}
        bpNIEADict = {}
        ccIEADict = {}
        ccNIEADict = {}
        mfIEADict = {}
        mfNIEADict = {}
        for gene in geneSet:
            bpIEADict[gene] = set()
            bpNIEADict[gene] = set()
            ccIEADict[gene] = set()
            ccNIEADict[gene] = set()
            mfIEADict[gene] = set()
            mfNIEADict[gene] = set()

        bpPath = './data/ontology/bp'
        ccPath = './data/ontology/cc'
        mfPath = './data/ontology/mf'
        bpTermSet = Ontology(bpPath).getTermSet()
        # bpAnceDict = Ontology(bpPath).getAncestorDict()
        ccTermSet = Ontology(ccPath).getTermSet()
        # ccAnceDict = Ontology(ccPath).getAncestorDict()
        mfTermSet = Ontology(mfPath).getTermSet()
        # mfAnceDict = Ontology(mfPath).getAncestorDict()
        for line in self.f:
            split = line.split('\t')
            if split[4] in bpTermSet:
                bpIEADict[split[2]].add(split[4])
            elif split[4] in ccTermSet:
                ccIEADict[split[2]].add(split[4])
            elif split[4] in mfTermSet:
                mfIEADict[split[2]].add(split[4])

        for line in self.f:
            split = line.split('\t')
            if split[4] in bpTermSet and split[6] != 'IEA':
                bpNIEADict[split[2]].add(split[4])
            elif split[4] in ccTermSet and split[6] != 'IEA':
                ccNIEADict[split[2]].add(split[4])
            elif split[4] in mfTermSet and split[6] != 'IEA':
                mfNIEADict[split[2]].add(split[4])
        """
        for gene in bpIEADict:
            for term in bpIEADict[gene]:
                bpIEADict[gene] = bpIEADict[gene].union(bpAnceDict[term])
        for gene in bpNIEADict:
            for term in bpNIEADict[gene]:
                bpNIEADict[gene] = bpNIEADict[gene].union(bpAnceDict[term])
        for gene in ccIEADict:
            for term in ccIEADict[gene]:
                ccIEADict[gene] = ccIEADict[gene].union(ccAnceDict[term])
        for gene in ccNIEADict:
            for term in ccNIEADict[gene]:
                ccNIEADict[gene] = ccNIEADict[gene].union(ccAnceDict[term])
        for gene in mfIEADict:
            for term in mfIEADict[gene]:
                mfIEADict[gene] = mfIEADict[gene].union(mfAnceDict[term])
        for gene in mfNIEADict:
            for term in mfNIEADict[gene]:
                mfNIEADict[gene] = mfNIEADict[gene].union(mfAnceDict[term])
        """

        return bpIEADict, bpNIEADict, ccIEADict, ccNIEADict, mfIEADict, mfNIEADict

    def getAnnotationDict_Ance(self):
        """
        包含祖先的注释字典，groupwise需要考虑
        Return:
        ----------------------------------------------------
        bpIEADict: IEA注释的bp
        bpNIEADict: NIEA注释的bp
        ccIEADict:IEA注释的cc
        ccNIEADict:NIEA注释的cc
        mfIEADict:IEA注释的mf
        mfNIEADict:NIEA注释的mf
        """
        geneSet = self.getGeneSet()
        bpIEADict = {}
        bpNIEADict = {}
        ccIEADict = {}
        ccNIEADict = {}
        mfIEADict = {}
        mfNIEADict = {}
        for gene in geneSet:
            bpIEADict[gene] = set()
            bpNIEADict[gene] = set()
            ccIEADict[gene] = set()
            ccNIEADict[gene] = set()
            mfIEADict[gene] = set()
            mfNIEADict[gene] = set()

        bpPath = './data/ontology/bp'
        ccPath = './data/ontology/cc'
        mfPath = './data/ontology/mf'
        bpTermSet = Ontology(bpPath).getTermSet()
        bpAnceDict = Ontology(bpPath).getAncestorDict()
        ccTermSet = Ontology(ccPath).getTermSet()
        ccAnceDict = Ontology(ccPath).getAncestorDict()
        mfTermSet = Ontology(mfPath).getTermSet()
        mfAnceDict = Ontology(mfPath).getAncestorDict()
        for line in self.f:
            split = line.split('\t')
            if split[4] in bpTermSet:
                bpIEADict[split[2]].add(split[4])
            elif split[4] in ccTermSet:
                ccIEADict[split[2]].add(split[4])
            elif split[4] in mfTermSet:
                mfIEADict[split[2]].add(split[4])

        for line in self.f:
            split = line.split('\t')
            if split[4] in bpTermSet and split[6] != 'IEA':
                bpNIEADict[split[2]].add(split[4])
            elif split[4] in ccTermSet and split[6] != 'IEA':
                ccNIEADict[split[2]].add(split[4])
            elif split[4] in mfTermSet and split[6] != 'IEA':
                mfNIEADict[split[2]].add(split[4])

        for gene in bpIEADict:
            for term in bpIEADict[gene]:
                bpIEADict[gene] = bpIEADict[gene].union(bpAnceDict[term])
        for gene in bpNIEADict:
            for term in bpNIEADict[gene]:
                bpNIEADict[gene] = bpNIEADict[gene].union(bpAnceDict[term])
        for gene in ccIEADict:
            for term in ccIEADict[gene]:
                ccIEADict[gene] = ccIEADict[gene].union(ccAnceDict[term])
        for gene in ccNIEADict:
            for term in ccNIEADict[gene]:
                ccNIEADict[gene] = ccNIEADict[gene].union(ccAnceDict[term])
        for gene in mfIEADict:
            for term in mfIEADict[gene]:
                mfIEADict[gene] = mfIEADict[gene].union(mfAnceDict[term])
        for gene in mfNIEADict:
            for term in mfNIEADict[gene]:
                mfNIEADict[gene] = mfNIEADict[gene].union(mfAnceDict[term])


        return bpIEADict, bpNIEADict, ccIEADict, ccNIEADict, mfIEADict, mfNIEADict






