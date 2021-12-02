import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns



#---- Get data by using numpy
data =np.genfromtxt('./../../02_clear_data/new_train.csv',delimiter=',')
data = np.delete(data, 0 ,0) #delete header
row, col = np.shape(data)
x = data[:,0:col-1]
y = data[:,col-1]
row, col = np.shape(x)

factor_name=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']


#------ Scaler
from sklearn.preprocessing import StandardScaler

#------- 5-fold cross validation
roc_score_svm       = 0
f1_score_weight_svm = 0
f1_score_macro_svm  = 0
f1_score_acc_svm    = 0
main_feature_svm = np.zeros((1, col))

roc_score       = 0
f1_score_weight = 0
f1_score_macro  = 0
f1_score_acc    = 0
main_feature = np.zeros((1, col))

fold_num = 5
kf = KFold(n_splits=fold_num)
n_fold = 0
for train_index, test_index in kf.split(x):
    n_fold = n_fold+1
    x_train_pre, x_test = x[train_index], x[test_index]
    y_train_pre, y_test = y[train_index], y[test_index]
    
    
    x_train, y_train = RandomOverSampler(random_state=1).fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = SMOTE().fit_resample(x_train_pre, y_train_pre)
    #x_train, y_train = ADASYN().fit_resample(x_train_pre, y_train_pre)
    
    #------ Scaler 
   # scaler = StandardScaler()
   # x_train_scaled = scaler.fit_transform(x_train)

    
    print("\n ============================================ PROCESS FOLD {}...".format(n_fold))
    train_num = len(x_train)
    test_num = len(x_test)
     

    print("\n ---------- Classifier: LogisticRegression, train {} and test {}".format(len(x_train), len(x_test)))
    classifier = LogisticRegression(C=0.1)
    classifier.fit(x_train, y_train)
    
    predicted = classifier.predict_proba(x_train)
    predicted = np.argmax(predicted, 1)
    expected  = y_train
    
    cm = metrics.confusion_matrix(expected, predicted)
    print("Training Confusion Matrix:\n{}".format(cm))
    print("Training Acc.={}".format(round(metrics.accuracy_score(expected, predicted),2) ))
    main_feature = main_feature + classifier.coef_
    
    predicted = classifier.predict_proba(x_test)
    predicted = np.argmax(predicted, 1)
    expected  = y_test

    cm = metrics.confusion_matrix(expected, predicted)
    print("\nTesting Confusion Matrix:\n{}".format(cm))
    print("Testing Acc.={}".format(round(metrics.accuracy_score(expected, predicted), 2) ))

    roc_score       = roc_score + roc_auc_score(expected, predicted)
    f1_score_weight = f1_score_weight + f1_score(expected, predicted, average='weighted')
    f1_score_macro  = f1_score_macro + f1_score(expected, predicted, average='macro')
    f1_score_acc    = f1_score_acc + metrics.accuracy_score(expected, predicted)

roc_score       = roc_score/fold_num
f1_score_weight = f1_score_weight/fold_num
f1_score_macro  = f1_score_macro/fold_num
f1_score_acc    = f1_score_acc/fold_num

roc_score_svm       = roc_score_svm/fold_num
f1_score_weight_svm = f1_score_weight_svm/fold_num
f1_score_macro_svm  = f1_score_macro_svm/fold_num
f1_score_acc_svm    = f1_score_acc_svm/fold_num

main_feature = main_feature/fold_num
main_feature_svm = main_feature_svm/fold_num

print('\n\n ================================== FINAL REPORT: Logistic Regresion...')
print(' +ROC_AUC_SCORE:{}\n +ACC:{}\n +F1_SCORE_WEIGHTED:{}\n +F1_SCORE_MACRO:{}\n'.format(roc_score, f1_score_acc, f1_score_weight, f1_score_macro))


feature_imp = pd.Series(classifier.coef_,index=factor_name).sort_values(ascending=False)
print(feature_imp)
sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()



