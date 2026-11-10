
import joblib

department_model = joblib.load("models/department_model.pkl")

test_complaints = [
    "The water pipe in the academic block classroom is leaking and flooding the floor. Please arrange a repair.",
    "The water dispenser near the lecture hall is empty every afternoon. Students need access to drinking water.",
    "The hostel bathroom has had no water supply for two days. Residents cannot use the washrooms.",
    "The main pipeline in the basement has burst and water is flooding the electrical room.",
    "The classroom projector is not working during lectures.",
    "The hostel washing machines have been broken for a week."
]

print("\n========== TOP 3 DEPARTMENT PREDICTIONS ==========\n")

for complaint in test_complaints:
    probabilities = department_model.predict_proba([complaint])[0]
    top_indices = probabilities.argsort()[::-1][:3]

    print("Complaint:", complaint)

    for index in top_indices:
        department = department_model.classes_[index]
        confidence = probabilities[index] * 100
        print(f"  {department}: {confidence:.2f}%")

    print("-" * 65)
