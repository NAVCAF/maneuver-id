import json
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare as friedman

def report_wilcoxon(baseline: list, models: list, names: list)->tuple:
    '''
    DESC
    ---
    This function creates wilcoxon test results for the baseline scores 
    with all implemented scores, and amongst all implemented scores
    ---
    INPUTS
    ---

    baseline: list of scores for the baseline implementation
    scores: list of lists pf scores of implementations
    names: list of strings with names of implementations
    ---
    OUTPUTS
    ---
    baseline_stats: (dict) dictionary of type 
                    {implementation_name : wilcoxon result with baseline}
    cross_stats: (dict) dictionary of type 
                {implementation_1_name : {implementation_2_name : wilcoxon result with implementation_1}

    '''
    baseline_stats = {}
    for name, model in zip(names, models):
        baseline_stats[name] = wilcoxon(model, baseline)
    
    cross_stats = {name:{} for name in names}
    for name_1, model_1 in zip(names, models):
        for name_2, model_2 in zip(names, models):
            if name_1 != name_2:
                cross_stats[name_1][name_2] = wilcoxon(model_1, model_2)
    
    return baseline_stats, cross_stats

def report_friedman(baseline, models):
    '''
    DESC
    ---
    This function creates friedmann_chi_squared test results for the baseline scores 
    with all implemented scores
    ---
    INPUTS
    ---

    baseline: list of scores for the baseline implementation
    scores: list of lists pf scores of implementations
    ---
    OUTPUTS
    ---
    freidmann stats: scipy.stats object with stat and p value

    '''
    all_models = [baseline] + models
    return friedman(*all_models)


def get_statistics(baseline: list, models: list, names: list, file_path = None):
    '''
    DESC
    ---
    This function creates friedmann_chi_squared test results for the baseline scores 
    with all implemented scores
    ---
    INPUTS
    ---

    baseline: list of scores for the baseline implementation
    scores: list of lists pf scores of implementations
    names: list of strings with names of implementations
    file_path: (str, None) path to location where statistics need to be saved
    ---
    OUTPUTS
    ---
    stat: dictionary with all statistical test stats

    '''
    baseline_wilcoxon_stats, cross_wilcoxon_stats = report_wilcoxon(baseline, models, names)
    friedman_stats = report_friedman(baseline,models)
    stat = {}
    stat["wilcoxon_baseline"] = baseline_wilcoxon_stats
    stat["wilcoxon_cross"] = cross_wilcoxon_stats
    stat["friedmann"] = friedman_stats
    
    try:
        with open(file_path, "w") as f:
            json.dump(stat, f, indent=2)
    except:
        pass
    
    return stat