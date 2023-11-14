from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict([X_test, X_non_test])
y_pred = y_pred.round()

# Convert one hot encoder to int 
y_test_arg = np.argmax(y_test, axis=1)
y_pred_arg = np.argmax(y_pred, axis=1)

# accuracy
acc = accuracy_score(y_test_arg, y_pred_arg)
print("Accuracy is: %.4f" %acc)

# AUC score
roc = roc_auc_score(y_test, y_pred,)
print("AUC is: %.2f" %roc)

# F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
f_1 = f1_score(y_test, y_pred, average='micro')
print(f"F-1 Score: {f_1:.2f}")

# Precision
precision = precision_score(y_test, y_pred, average='micro')
print(f"Precision: {precision:.2f}")

# Confusion matrix
confusion_matrix = confusion_matrix(y_test_arg, y_pred_arg)
print(confusion_matrix)

# TP FP TN FN Values
tp = np.diagonal(confusion_matrix)
fp = confusion_matrix.sum(axis=0) - tp
fn = confusion_matrix.sum(axis=1) -tp
tn = confusion_matrix.sum() - (tp+fp+fn)

# Specificity
specificity_values = tn / (tn + fp)
sp = np.mean(specificity_values)

# False Positive Rate
fpr_values = fp / (fp + tn)
fpr = np.mean(fpr_values)

# Print all the results using PrettyTable
from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Metric", "Result"]
table.add_row(["Acc", f"{acc:.4f}"])
table.add_row(["AUC", f"{roc:.4f}"])
table.add_row(["F1", f"{f_1:.4f}"])
table.add_row(["PPV", f"{precision:.4f}"])
table.add_row(["FPR", f"{fpr:.4f}"])
table.add_row(["SP", f"{sp:.4f}"])
print(table)
