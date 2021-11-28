import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tru_pos= np.sum(np.logical_and(prediction == ground_truth, prediction == True)) 
    all_pos= np.sum(prediction)
    gr_tru=np.sum(ground_truth)
    all_pred=len(prediction)
    corect_pred=np.sum(prediction == ground_truth) 

    precision = tru_pos/(all_pos)
    recall = tru_pos/gr_tru
    accuracy = corect_pred/all_pred
    f1 = 2 * precision*recall/(precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    corect_pred=np.sum(prediction == ground_truth) 
    all_pred=len(prediction)
    accuracy = corect_pred/all_pred
    return accuracy
