from audioop import cross
from email.mime import base
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare as friedman

def report_wilcoxon(baseline: list, scores: list, names: list)->tuple:
    baseline_stats = {}
    for name, score in zip(names, scores):
        baseline_stats[name] = wilcoxon(score, baseline)
    
    cross_stats = {name:{} for name in names}
    for name_1, score_1 in zip(names, scores):
        for name_2, score_2 in zip(names, scores):
            if name_1 != name_2:
                cross_stats[name_1][name_2] = wilcoxon(score_1, score_2)
    
    return baseline_stats, cross_stats

def report_friedman(baseline, scores):
    all_scores = [baseline] + scores
    return friedman(all_scores)