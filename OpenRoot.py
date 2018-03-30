#!/usr/bin/env python3
# encoding: utf-8

# File        : OpenRoot.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2018 Mar 17
#
# Description : 

import rootpy
from rootpy.io import root_open
import ROOT
import pprint
import numpy as np
from collections import OrderedDict

def OpenFile(filename):
    file = root_open(filename, 'read')
    return file

def GetTree(file, treename):
    ROOT.gROOT.ProcessLine("".join(open("L1AnalysisL1UpgradeDataFormat.h", 'r').readlines()))
    l1ntuple = ROOT.L1Analysis.L1AnalysisL1UpgradeDataFormat()
    g = file.Get(treename)
    g.SetBranchAddress("L1Upgrade", l1ntuple)
    return g, l1ntuple

def Parsentuple(ntuple):
    jets =[]
    for i, bx in enumerate(ntuple.jetBx):
        if bx != 0:
            continue
        jets += [ntuple.jetEt[i], ntuple.jetEta[i], ntuple.jetPhi[i]]
        # print(ntuple.jetEt[i], ntuple.jetEta[i], ntuple.jetPhi[i],
              # ntuple.jetRawEt[i], ntuple.jetSeedEt[i], ntuple.jetPUEt[i],
              # ntuple.jetPUDonutEt0[i], ntuple.jetPUDonutEt1[i],
              # ntuple.jetPUDonutEt2[i],  ntuple.jetPUDonutEt3[i])
    return jets

def StoreNumpy(tree, ntuple, outname):
    entries = tree.GetEntries()
    jetarray = np.zeros((entries, 12*3))
    for i in xrange(entries):
        tree.GetEntry(i)
        jets =Parsentuple(ntuple)
        temp = np.empty(12*3)
        temp[:len(jets)]  = jets
        jetarray[i] = temp
    np.save("%s.npy" % outname, jetarray)

def StoreJets(upntuple, emntuple):
    upjets = []
    emjets = []
    jetmap =OrderedDict()

    for i, bx in enumerate(upntuple.jetBx):
        if bx != 0:
            continue
        upjets.append((upntuple.jetEt[i], upntuple.jetEta[i], upntuple.jetPhi[i]))
        # print(ntuple.jetEt[i], ntuple.jetEta[i], ntuple.jetPhi[i],
    for i, bx in enumerate(emntuple.jetBx):
        if bx != 0:
            continue
        emjets.append((emntuple.jetEt[i], emntuple.jetEta[i], emntuple.jetPhi[i]))
    if len(upjets) != len(emjets):
        print("Different number of jets: unpack %s emulated %s" % (len(uptree), len(emjets)))

    # print(upjets, emjets)
    for i in upjets:
        for j in emjets:
            if SameObject(i, j):
                jetmap[i] = j
                break
        if i not in jetmap:
            jetmap[i]= (0, 0, 0)
    return jetmap

def SameObject(i, j):
    if ((i[1] -j[1])**2 + (i[2] -j[2])**2 ) > 0.4**2:
        return False
    if j[0] == 0:
        return i[0] ==0
    elif abs(i[0]/j[0]-1) > 0.5:
        return False
    return True


if __name__ == "__main__":
    file = OpenFile("L1Ntuple_94.root")
    uptree, upntuple = GetTree(file, "l1UpgradeTree/L1UpgradeTree")
    emtree, emntuple = GetTree(file, "l1UpgradeEmuTree/L1UpgradeTree")
    entries = uptree.GetEntries()

    upjets = []
    emjets = []
    # emjets = np.empty(3)
    for i in xrange(entries):
        uptree.GetEntry(i)
        emtree.GetEntry(i)
        js = StoreJets(upntuple, emntuple)
        upjets += list(js.keys())
        emjets += list(js.values())
        # print js, list(js.keys())
        # np.append(upjets, u, axis=0)
        # np.append(emjets, e, axis=0)
    # pprint.pprint( np.asarray(upjets))


    np.save("unpack.npy" , np.asarray(upjets))
    np.save("emulator.npy", np.asarray(emjets))
    #StoreNumpy(uptree, upntuple, "unpack")
    #StoreNumpy(emtree, emntuple, "emulator")
