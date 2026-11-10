
import spacy
import re

nlp = spacy.load("en_core_web_sm")

campus_locations = [
    "Block A", "Block B", "Block C", "Block D",
    "Block E", "Library", "Main Gate", "Admin Block",
    "Academic Block", "Hostel", "Auditorium",
    "Cafeteria", "Parking Area", "Computer Lab",
    "Science Lab", "Sports Ground"
]


def extract_entities(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "entity": ent.text.strip(),
            "type": ent.label_,
            "description": spacy.explain(ent.label_)
        })

    for location in campus_locations:

        pattern = r"\b" + re.escape(location) + r"\b"

        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:

            entities.append({
                "entity": match.group(),
                "type": "CAMPUS_LOCATION",
                "description": "Recognised campus location"
            })

    unique_entities = []
    seen = set()

    for entity in entities:

        key = (entity["entity"].lower(), entity["type"])

        if key not in seen:
            unique_entities.append(entity)
            seen.add(key)

    return unique_entities


if __name__ == "__main__":

    complaint = """
    The water leakage in Block A hostel has continued
    since Monday. I reported it to the Maintenance
    Department at 9 AM, but nobody has visited yet.
    """

    print("\n========== CAMPUSPULSE ENTITY EXTRACTION ==========\n")

    results = extract_entities(complaint)

    for entity in results:

        print("Entity:", entity["entity"])
        print("Type:", entity["type"])
        print("Meaning:", entity["description"])
        print("-" * 50)
