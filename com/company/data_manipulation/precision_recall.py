import    numpy    as    np
from    sklearn.metrics    import    precision_recall_curve
y_true = np.array([0, 1, 1, 0, 1])
y_scores = np.array([0.5, 0.6, 0.38, 0.9, 1])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print    precision
print    recall
