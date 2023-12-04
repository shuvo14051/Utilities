# Precision recall curve
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
import numpy as np
import tensorflow as tf
from tensorflow import keras

# When we have array like [0,1,2,3,4,1,2,3]
# Either use the following function or one hot encode the values and use the regular function from sklearn
"""
This is the code for creating one hot encoding formation
data = y_test.copy()
data2 = y_pred.copy()

from sklearn.preprocessing import OneHotEncoder
data = data.reshape(-1, 1)
data2 = data2.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False, categories='auto')
y_test_hot = encoder.fit_transform(data)
y_pred_hot = encoder.fit_transform(data2)
"""
def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

def pr_curve(model_path):
    model = tf.keras.models.load_model(model_path)
    y_scores = model.predict([X_test, X_non_test])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    classifier = OneVsRestClassifier(model)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(4):  # Assuming you have 4 classes
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_scores[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='Class {0} (AP = {1:0.2f})'.format(i, average_precision[i]))

    # Plot the micro-averaged precision-recall curve
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_test_bin.ravel(), y_scores.ravel())
    average_precision["macro"] = average_precision_score(y_test_bin, y_scores, average="macro")
    # plt.plot(recall["macro"], precision["macro"], label='Macro-average Precision-Recall Curve (AP = {0:0.2f})'.format(average_precision["macro"]), color='deeppink', linestyle=':')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Multi-Class')
    plt.legend()
    plt.show()


# ROC Curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_curve():
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 5))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (class {0}) (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
