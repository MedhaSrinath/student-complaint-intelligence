import joblib, json, sys
artifact = joblib.load('experiments/category_classification/logreg_model.pkl')
model = artifact['model']
print('\n'.join(model.classes_))
