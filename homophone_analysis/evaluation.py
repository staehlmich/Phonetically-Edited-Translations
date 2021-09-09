#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Michael Staehli

import typing
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def prf(tp, fp, fn):
    try:
        prec = tp / (tp+fp)
    except ZeroDivisionError:
        prec = 1  # no FP -- perfect precision
    try:
        recall = tp / (tp+fn)
    except ZeroDivisionError:
        recall = 1  # no FN -- perfect recall
    try:
        f1 = 2*prec*recall / (prec+recall)
    except ZeroDivisionError:
        f1 = 0
    return prec, recall, f1

class MicroManager:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def average(self):
        return prf(self.tp, self.fp, self.fn)

class MacroManager:
    def __init__(self):
        self.prec = []
        self.recall = []
        self.f1 = []

    def update(self, tp, fp, fn):
        prec, recall, f1 = prf(tp, fp, fn)
        self.prec.append(prec)
        self.recall.append(recall)
        self.f1.append(f1)

    def average(self):
        return tuple(sum(m)/len(m) for m in (self.prec, self.recall, self.f1))
#Alternative: from sklearn import metrics
#precision = metrics.precision_score(true_labels, pred_labels, pos_label=label)

def evaluate(gold:str, test:str):
    micro = MicroManager()
    macro = MacroManager()
    for ground_truth, predictions in read_docs(gold, test):
        print(ground_truth, predictions)
        tp, fp, fn = get_elements(ground_truth, predictions)
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        micro.update(tp, fp, fn)
        macro.update(tp, fp, fn)
    return micro.average(), macro.average()

def read_prediction(line:str) -> list:
    """
    Helper function to read a line from a gold orprediction file and
    return a list with pet matches.
    @param line:
    @return: List with pet matches [source, target] or ["NA", "NA"]
    """
    pets = [elem.split("|||") for elem in line.rstrip().split("\t")]
    if pets != [[""]]:
        return pets
    else:
        return []

def read_docs(gold:str, test:str):
    """
    Function to open test file and gold file and read phonetically
    edited translations.
    @param gold: File containing gold pets from test file.
    @param test: File containing prediction pets from test file.
    @return: dics containing gold-pets and pred-pets.
    """
    with open(gold, "r", encoding="utf-8") as gld, \
    open(test, "r", encoding="utf-8") as tst:
        for truth, pred in zip(gld, tst):
            truth = {src:trg for src, trg in read_prediction(truth)}
            #Hack to get only necessary columns
            pred = {pet[0]:pet[2] for pet in read_prediction(pred)}
            yield truth, pred

def get_elements(truth: dict, pred:dict):
    """
    Helper function that compares two lists with pets and returns elements
    to calculate precision, recall and f-measure.
    @param truth: dict with gold-pets of a sentence.
    @param pred: dict with prediction pets of a sentence.
    @return:
    """
    elems = {"tp":0, "fp": 0, "fn": 0}
    for src, trg in pred.items():
        for gld_src in truth:
            if src in gld_src:
                #Check if any of the translations in gold_trans.
                if any(trans in truth[gld_src] for trans in trg.split("/")) \
                    == True:
                    elems["tp"] += 1
    #Check false positives.
    for src in pred:
        if any(src in gld_src for gld_src in truth) == False:
            elems["fp"] += 1
    #Check false negatives.
    for gld_src in truth:
        if any(src in gld_src for src in pred) == False:
            elems["fn"] +=1

    return elems["tp"], elems["fp"], elems["fn"]

def main():
    micro, macro = evaluate("gold_pets.txt", "evaluation_pets2.txt")

if __name__ == "__main__":
    main()