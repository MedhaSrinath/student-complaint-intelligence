import joblib, json, sys
path = r'C:\\RVU\\Sem5\\student-complaint-intelligence\\experiments\\category_classification\\logreg_model.pkl'
artifact = joblib.load(path)
print('artifact type:', type(artifact))
if isinstance(artifact, dict):
    print('keys:', list(artifact.keys()))
    model = artifact.get('model')
    print('inner model type:', type(model))
    print('has predict_proba?', hasattr(model, 'predict_proba'))
    print('has classes_?', hasattr(model, 'classes_'))
else:
    model = artifact
    print('model type:', type(model))
    print('has predict_proba?', hasattr(model, 'predict_proba'))
    print('has classes_?', hasattr(model, 'classes_'))
