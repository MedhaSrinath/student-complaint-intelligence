import joblib, os, sys
path = r"C:\\RVU\\Sem5\\student-complaint-intelligence\\experiments\\category_classification\\svm_model.pkl"
artifact = joblib.load(path)
print('Artifact type:', type(artifact))
if isinstance(artifact, dict):
    print('Keys:', list(artifact.keys()))
    model = artifact.get('model')
    print('Model type:', type(model))
    print('Has predict_proba?', hasattr(model, 'predict_proba'))
    print('Has classes_?', hasattr(model, 'classes_'))
else:
    model = artifact
    print('Model type:', type(model))
    print('Has predict_proba?', hasattr(model, 'predict_proba'))
    print('Has classes_?', hasattr(model, 'classes_'))
