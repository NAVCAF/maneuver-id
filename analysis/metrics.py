from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch 
import matplotlib.pyplot as plt
import seaborn as sns

def get_predictions(model, test_loader):
    """
    DESC
    ---
    Calculates predictions and corresponding softmax logits 
    for the given data to be used for printing metrics
    ---
    INPUTS
    ---
    Model - Model used for evaluation
    Test Dataloader - Dataloader for test data
    ---
    RETURN
    ---
    y_true - ground truth labels
    y_pred - model predictions
    y_pred_prob - logits of model predictions
    """
    y_true = []
    y_pred = []
    y_pred_prob =[]
    for x, y, lx in test_loader:
        model.eval()
        pred = model(x.cuda())
        prob = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(prob, dim=1)
        y_pred_prob.extend(prob.cpu().detach().numpy()[:,1])
        y_pred.extend(list(pred.cpu().detach().numpy()))
        y_true.extend(list(y.cpu().detach().numpy()))    

    return y_true, y_pred, y_pred_prob

def plot_confusion_matrix(y_pred, y_true):
    """
    DESC
    ---
    Calculates confusion matrix values, prints the false positive and negative rates,
    true positive and negative rates, and plots the heatmap for the confusion matrix
    ---
    INPUTS
    ---
    y_pred - model predictions 
    y_true - ground truth labels
    ---
    RETURN
    ---
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[0]
    fn, tn = cm[1]
    fpr = fp / (tn + fp)
    tpr = tp / (tp + fn)
    fnr = fn / (tp + fn)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion matrix '); 
    ax.xaxis.set_ticklabels(['Bad', 'Good']); ax.yaxis.set_ticklabels(['Bad', 'Good'])
    plt.show()

    print("TPR:", tpr, "FPR:", fpr, "FNR:", fnr, "TNR:", (1-fpr))

def get_metrics(y_pred, y_true):
    """
    DESC
    ---
    Creates the classification report that includes
    precision, recall and F1 score for the model
    ---
    INPUTS
    ---
    y_pred - model predictions 
    y_true - ground truth labels
    ---
    RETURN
    ---
    None
    """
    print(classification_report(y_true, y_pred))


def get_roc_auc(y_pred_prob, y_true, show_plot=False):
    """
    DESC
    ---
    Computes the roc-auc score for given predictions and optionally
    plots the curve
    ---
    INPUTS
    ---
    y_pred_prob - predicted logits
    y_true - ground truth labels
    show_plot - optional parameter to show plots
    ---
    RETURN
    ---
    roc - roc-auc score for the model
    """
    # Plot ROC curve
    if show_plot == True:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    roc = roc_auc_score(y_true, y_pred_prob)
    
    return roc