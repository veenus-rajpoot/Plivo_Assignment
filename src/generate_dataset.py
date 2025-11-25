import random
import json
import uuid

# ------------ Helpers ------------ #
DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

FIRST_NAMES = ["rahul", "mohit", "priya", "arjun", "neha", "veer", "riya", "sachin"]
LAST_NAMES = ["kumar", "sharma", "singh", "rajpoot", "patel", "yadav", "verma"]

CITIES = ["mumbai", "delhi", "kolkata", "chennai", "pune", "lucknow"]
LOCATIONS = ["aiims hospital", "bandra station", "marine drive", "iit bombay"]

def spell_number(num_string):
    return " ".join(DIGITS[d] for d in num_string)

def make_phone():
    digits = "".join(str(random.randint(0,9)) for _ in range(10))
    text = spell_number(digits)
    return digits, text

def make_credit_card():
    digits = "".join(str(random.randint(0,9)) for _ in range(16))
    text = spell_number(digits)
    return digits, text

def make_email():
    name = random.choice(FIRST_NAMES)
    domain = random.choice(["gmail", "yahoo", "hotmail"])
    full = f"{name}@{domain}.com"
    noisy = f"{name} at {domain} dot com"
    return full, noisy

def make_date():
    year = random.randint(1990, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    clean = f"{day}-{month}-{year}"
    noisy = f"{spell_number(str(day))} {spell_number(str(month))} {spell_number(str(year))}"
    return clean, noisy


# ------------ Generator ------------ #
def generate(n_samples, out_path):
    samples = []

    for _ in range(n_samples):
        template = random.choice([
            "my phone number is {PHONE}",
            "contact me at {EMAIL}",
            "my credit card number is {CREDIT}",
            "my name is {NAME} and I live in {CITY}",
            "the location is {LOCATION}",
            "the date is {DATE}"
        ])

        text = template
        entities = []

        # PHONE
        if "{PHONE}" in text:
            clean, noisy = make_phone()
            start = len(template.split("{PHONE}")[0])
            end = start + len(noisy)
            entities.append({"start": start, "end": end, "label": "PHONE"})
            text = text.replace("{PHONE}", noisy)

        # EMAIL
        if "{EMAIL}" in text:
            clean, noisy = make_email()
            start = len(template.split("{EMAIL}")[0])
            end = start + len(noisy)
            entities.append({"start": start, "end": end, "label": "EMAIL"})
            text = text.replace("{EMAIL}", noisy)

        # CREDIT CARD
        if "{CREDIT}" in text:
            clean, noisy = make_credit_card()
            start = len(template.split("{CREDIT}")[0])
            end = start + len(noisy)
            entities.append({"start": start, "end": end, "label": "CREDIT_CARD"})
            text = text.replace("{CREDIT}", noisy)

        # NAME
        if "{NAME}" in text:
            name = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
            start = len(template.split("{NAME}")[0])
            end = start + len(name)
            entities.append({"start": start, "end": end, "label": "PERSON_NAME"})
            text = text.replace("{NAME}", name)

        # CITY
        if "{CITY}" in text:
            city = random.choice(CITIES)
            start = len(text.split("{CITY}")[0])
            end = start + len(city)
            entities.append({"start": start, "end": end, "label": "CITY"})
            text = text.replace("{CITY}", city)

        # LOCATION
        if "{LOCATION}" in text:
            loc = random.choice(LOCATIONS)
            start = len(text.split("{LOCATION}")[0])
            end = start + len(loc)
            entities.append({"start": start, "end": end, "label": "LOCATION"})
            text = text.replace("{LOCATION}", loc)

        # DATE
        if "{DATE}" in text:
            clean, noisy = make_date()
            start = len(template.split("{DATE}")[0])
            end = start + len(noisy)
            entities.append({"start": start, "end": end, "label": "DATE"})
            text = text.replace("{DATE}", noisy)

        samples.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "entities": entities
        })

    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ------------ Run ------------ #
generate(1000, "data/train.jsonl")
generate(200, "data/dev.jsonl")
print("Done! Generated train and dev sets.")
