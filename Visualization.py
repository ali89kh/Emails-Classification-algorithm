# 5-1 Confusion matrix
# 5-1-1 Logistic Regression
plt.figure()
ax = sns.heatmap(LR_cm, annot=True, cmap = 'Blues')
ax = ax.set(xlabel='Predicted',ylabel='True',title='Logistic Regression - Confusion Matrix',
            xticklabels=(directories),
            yticklabels=(directories))

plt.savefig('Logistic_Regression_Confusion_Matrix.png', dpi = 300)

# 5-1-2 Random Forest
plt.figure()
ax = sns.heatmap(RF_cm, annot=True, cmap = 'Blues')
ax = ax.set(xlabel='Predicted',ylabel='True',title='Random Forest - Confusion Matrix',
            xticklabels=(directories),
            yticklabels=(directories))

plt.savefig('Random_Forest_Confusion_Matrix.png', dpi = 300)

# 5-1-3 Naive Bayes
plt.figure()
ax = sns.heatmap(gnb_cm, annot=True, cmap = 'Blues')
ax = ax.set(xlabel='Predicted',ylabel='True',title='Naive Bayes - Confusion Matrix',
            xticklabels=(directories),
            yticklabels=(directories))

plt.savefig('Naive_Bayes_Confusion_Matrix.png', dpi = 300)

# 5-1-4 ANN
plt.figure()
ax = sns.heatmap(ann_cm, annot=True, cmap = 'Blues')
ax = ax.set(xlabel='Predicted',ylabel='True',title='ANN- Confusion Matrix',
            xticklabels=(directories),
            yticklabels=(directories))

plt.savefig('Artificial_Neural_Networks_Confusion_Matrix.png', dpi = 300)

