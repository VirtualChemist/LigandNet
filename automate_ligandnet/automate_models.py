import os
import sys
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

# parameters
DATA_DIR = "/data2/mhassan/Pharos"
OUTPUT_DIR = "output"
if not os.path.isdir(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

pname = "Caspase_1"
classifier_loglevel = 0
gridsearch_loglevel = 2

print("READING ACTIVES AND DECOYS")
actives = pd.read_csv(os.path.join(DATA_DIR, 'actives/fingerprints/Newdata_set2/'+pname+'.csv'), header=None)
decoys = pd.read_csv(os.path.join(DATA_DIR, 'decoys/fingerprints/Newdata_set2/decoys_'+pname+'.csv'), header=None)

print("Number of actives: {}, and number of decoys: {}".format(len(actives.index), len(decoys.index)))

# Train the protein model only if there are more than 50 samples
if len(actives.index) < 50:
  print("NUMBER OF SAMPLES IS LESS THAN 50. EXITING")
  exit()

# Start training
print("TRAINING FOR ", pname)

# Get the score and output path, if not found then create one
score_path = os.path.join(DATA_DIR, "scores/part2")
if not os.path.isdir(score_path):
    os.makedirs(score_path)
output = open(os.path.join(score_path, pname+'_score.csv'),'w')

actives_x = actives.iloc[:,1:].values
actives_y = np.ones(len(actives_x))

decoys_x = decoys.iloc[:,:].values
decoys_y = np.zeros(len(decoys_x))

print("PROCESSING TRAINING DATA")
x = np.concatenate((actives_x, decoys_x))
y = np.concatenate((actives_y, decoys_y)) #labels

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =1)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

print("BUILING CLASSIFIERS")

# Scoring metrics
scoring = {'auc_score': 'roc_auc', 
           'precision_score': make_scorer(precision_score),
           'recall_score': make_scorer(recall_score),
           'accuracy_score': make_scorer(accuracy_score)
          }

# Stratified K-fold
cross_validation_count = 5
skf = StratifiedKFold(n_splits=cross_validation_count)

# Random Forest
classifier_rf = RandomForestClassifier(class_weight='balanced', verbose=classifier_loglevel)
parameters_rf = {'n_estimators':[i for i in range(100, 1000, 50)]}
gridsearch_rf = GridSearchCV(classifier_rf, parameters_rf, scoring=scoring, cv=skf, refit='auc_score', n_jobs=-1, verbose=gridsearch_loglevel)
print("TRAINING THE RANDOM FOREST CLASSIFIER")
gridsearch_rf.fit(x_train, y_train)

# Support Vector Machine
classifier_sv = SVC(class_weight='balanced', kernel='linear', random_state=1, verbose=classifier_loglevel)
parameters_sv = {'C': [0.1, 1.0, 10, 100, 1000], 'gamma':[0.1, 1, 10, 100, 1000, 'auto']}
gridsearch_sv = GridSearchCV(classifier_sv, parameters_sv, scoring=scoring, cv=skf, refit='auc_score', n_jobs=-1, verbose=gridsearch_loglevel)
print("TRAINING THE SUPPORT VECTOR CLASSIFIER")
gridsearch_sv.fit(x_train, y_train)

# Neural Network
def get_hidden_layers():
    import itertools
    x = [64, 128, 256]
    hl = []
    
    for i in range(1, len(x)):
        hl.extend([p for p in itertools.product(x, repeat=i+1)])
    
    return hl

classifier_nn = MLPClassifier(solver='adam', alpha=1e-5, early_stopping=True, random_state=1, verbose=classifier_loglevel)
hidden_layer_sizes = get_hidden_layers()
parameters_nn = {'hidden_layer_sizes': hidden_layer_sizes}
gridsearch_nn = GridSearchCV(classifier_nn, parameters_nn, scoring=scoring, cv=skf, refit='auc_score', n_jobs=-1, verbose=gridsearch_loglevel)
print("TRAINING THE NEURAL NETWORK CLASSIFIER")
gridsearch_nn.fit(x_train, y_train)

# Get the best score and save the best model
classifiers = [gridsearch_rf.best_estimator_, gridsearch_sv.best_estimator_, gridsearch_nn.best_estimator_]
classifier_names = ['Random Forest', 'Support Vector Machine', "Neural Network"]
classifier_scores = [gridsearch_rf.best_score_, gridsearch_sv.best_score_, gridsearch_nn.best_score_]
best_classifier_index = classifier_scores.index(max(classifier_scores))

# Get the report
model = classifiers[best_classifier_index]
y_pred = model.y_pred(y_test)
classification_report = classification_report(y_test, y_pred)
result = {'protein': pname, 'best_classifier': classifier_names[best_classifier_index], 'best_score': classifier_scores[best_classifier_index], 'report': classification_report}

print("===================RESULT=====================")
print("PROTEIN NAME: ".format(pname))
print("BEST CLASSIFIER: {}".format(classifiers[best_classifier_index]))
print("BEST SCORE: {}".format(classifier_scores[best_classifier_index]))

print("SAVING THE RESULTS", end='...')

# Save the results
with open(os.path.join(OUTPUT_DIR, "result_" + pname), "wb") as f:
    pickle.dump(result, f)

filename = os.path.join(OUTPUT_DIR, "classifier_" + pname)
joblib.dump(classifiers[best_classifier_index], filename)
print("DONE")
