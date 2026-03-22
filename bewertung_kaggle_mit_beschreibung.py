import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json


# --------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------

def yn_to_de(value: str) -> str:
    """Konvertiert yes/no -> Ja/Nein für die Ausgabe."""
    value = value.strip().lower()
    if value == "yes":
        return "Ja"
    if value == "no":
        return "Nein"
    return value  # falls etwas anderes eingegeben wurde


def normalize_yes_no(value: str) -> str:
    """Normalisiert ja/yes/nein/no/y/n -> 'yes' oder 'no'."""
    value = value.strip().lower()
    if value in ["ja", "yes", "y", "j"]:
        return "yes"
    if value in ["nein", "no", "n"]:
        return "no"
    return value


# --------------------------------------------------
# 1. Preismodell auf dem Kaggle-Datensatz trainieren
# --------------------------------------------------

def train_price_model():
    # CSV wie aus dem Browser gespeichert
    data = pd.read_csv("kaggle_housing.csv")

    # Zielvariable
    y = data["price"]

    # Features laut Kaggle
    X = data[
        [
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking",
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
        ]
    ]

    # Kategorische Variablen in Dummy-Spalten umwandeln
    X = pd.get_dummies(X)

    # Train/Test-Split (für realistische Einschätzung)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, X.columns


# --------------------------------------------------
# 2. Preis für neues Objekt schätzen (Kaggle-Features)
# --------------------------------------------------

def estimate_price(model, feature_columns, input_data):
    row = pd.DataFrame(
        [[
            input_data["area"],
            input_data["bedrooms"],
            input_data["bathrooms"],
            input_data["stories"],
            input_data["parking"],
            input_data["mainroad"],
            input_data["guestroom"],
            input_data["basement"],
            input_data["hotwaterheating"],
            input_data["airconditioning"],
            input_data["prefarea"],
            input_data["furnishingstatus"],
        ]],
        columns=[
            "area",
            "bedrooms",
            "bathrooms",
            "stories",
            "parking",
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
        ],
    )

    # dieselbe One-Hot-Encoding-Logik wie beim Training
    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    price = model.predict(row)[0]
    return price


# --------------------------------------------------
# 3. Deutsche Beschreibung generieren (regelbasiert)
# --------------------------------------------------

def generate_description(input_data, price: float):
    ort = input_data["ort"]
    objektart = input_data["objektart"]
    zimmer = input_data["bedrooms"]
    wohnflaeche = input_data["area"]
    badezimmer = input_data["bathrooms"]
    stories = input_data["stories"]
    parking = input_data["parking"]
    furnishingstatus = input_data["furnishingstatus"]
    mainroad = input_data["mainroad"]
    prefarea = input_data["prefarea"]
    guestroom = input_data["guestroom"]
    basement = input_data["basement"]
    hotwaterheating = input_data["hotwaterheating"]
    airconditioning = input_data["airconditioning"]

    # Titel – Objektart dynamisch
    titel = f"Attraktive Immobilie ({objektart}) mit {zimmer} Zimmern in {ort.title()}"

    # Kurzbeschreibung – Objektart erwähnt, ohne Grammatikstress
    kurzbeschreibung = (
        f"Angeboten wird eine gepflegte Immobilie ({objektart.lower()}) mit ca. "
        f"{int(wohnflaeche)} m² Wohnfläche, {zimmer} Schlafzimmern und "
        f"{badezimmer} Badezimmer(n). Das Objekt verfügt über {stories} Etage(n) "
        f"und {parking} Parkplatz/Parkplätze."
    )

    # Möblierungstext
    if furnishingstatus == "furnished":
        ausstattung = "voll möbliert"
    elif furnishingstatus == "semi-furnished":
        ausstattung = "teilmöbliert"
    else:
        ausstattung = "unmöbliert"

    kurzbeschreibung += f" Die Ausstattung ist {ausstattung}."

    # Highlights-Liste
    highlights = [
        f"{int(wohnflaeche)} m² Wohnfläche",
        f"{zimmer} Schlafzimmer",
        f"{badezimmer} Badezimmer",
        f"{stories} Etage(n)",
    ]

    if parking > 0:
        highlights.append(f"{parking} Parkplatz/Parkplätze")

    if mainroad == "yes":
        highlights.append("Gute Erreichbarkeit durch Nähe zur Hauptstraße")
    else:
        highlights.append("Ruhigere Lage abseits der Hauptstraße")

    if guestroom == "yes":
        highlights.append("Separates Gästezimmer")
    if basement == "yes":
        highlights.append("Keller vorhanden")
    if hotwaterheating == "yes":
        highlights.append("Warmwasserheizung")
    if airconditioning == "yes":
        highlights.append("Klimaanlage")

    # Lage-Text
    if prefarea == "yes":
        lage_text = (
            f"Das Objekt liegt in einer bevorzugten Wohnlage von {ort.title()} "
            f"mit guter Infrastruktur und attraktiver Umgebung."
        )
    else:
        lage_text = (
            f"Die Lage in {ort.title()} bietet eine solide Infrastruktur "
            f"und eine gute Anbindung an das Umland."
        )

    # Preis formatiert & numerisch
    preis_euro = round(price, 2)
    preis_formatiert = f"{round(price):,} €".replace(",", ".")

    beschreibung = {
        "titel": titel,
        
        "kurzbeschreibung": kurzbeschreibung,
        "highlights": highlights,
        "lage_und_markt": lage_text,
        "geschaetzter_preis_euro": preis_euro,
        "geschaetzter_preis": preis_formatiert,
    }

    return beschreibung


# --------------------------------------------------
# 4. Hauptprogramm: Eingaben + JSON-Ausgabe
# --------------------------------------------------

def main():
    print(">>> Kaggle-Preismodell wird trainiert...")
    model, feature_columns = train_price_model()
    print("Fertig.\n")

    print("--- Bitte Immobiliendaten eingeben ---")
    input_data = {}

    # Basisdaten für Beschreibung & Kontakt
    input_data["strasse"] = input("Straße: ")
    input_data["hausnummer"] = input("Hausnummer: ")
    input_data["ort"] = input("Ort (z.B. Berlin): ")
    input_data["objektart"] = input(
        "Objektart (z.B. Wohnung / Einfamilienhaus / Mehrfamilienhaus / Büro): "
    ).strip()
    input_data["telefon"] = input("Telefonnummer: ")
    input_data["email"] = input("E-Mail: ")

    # Kaggle-Features (müssen zu den Spalten passen!)
    input_data["area"] = float(input("Wohnfläche (area) in m²: "))
    input_data["bedrooms"] = int(input("Schlafzimmer (bedrooms): "))
    input_data["bathrooms"] = int(input("Badezimmer (bathrooms): "))
    input_data["stories"] = int(input("Etagen (stories): "))
    input_data["parking"] = int(input("Parkplätze (parking): "))

    input_data["mainroad"] = normalize_yes_no(
        input("An Hauptstraße? (ja/nein): ")
    )
    input_data["guestroom"] = normalize_yes_no(
        input("Gästezimmer? (ja/nein): ")
    )
    input_data["basement"] = normalize_yes_no(
        input("Keller? (ja/nein): ")
    )
    input_data["hotwaterheating"] = normalize_yes_no(
        input("Warmwasserheizung? (ja/nein): ")
    )
    input_data["airconditioning"] = normalize_yes_no(
        input("Klimaanlage? (ja/nein): ")
    )
    input_data["prefarea"] = normalize_yes_no(
        input("Bevorzugte Lage? (ja/nein): ")
    )
    input_data["furnishingstatus"] = input(
        "Möblierung (furnished / semi-furnished / unfurnished): "
    ).strip()

    # Preis schätzen
    price = estimate_price(model, feature_columns, input_data)

    # Beschreibung erzeugen
    beschreibung = generate_description(input_data, price)

    # JSON-Antwort bauen
    response = {
        "titel": beschreibung["titel"],
        "kurzbeschreibung": beschreibung["kurzbeschreibung"],
        "highlights": beschreibung["highlights"],
        "lage_und_markt": beschreibung["lage_und_markt"],
        "geschaetzter_preis": beschreibung["geschaetzter_preis"],
        "geschaetzter_preis_euro": beschreibung["geschaetzter_preis_euro"],
        "objektdaten": {
            "adresse": f"{input_data['strasse'].strip()} "
                       f"{input_data['hausnummer'].strip()}, "
                       f"{input_data['ort'].title().strip()}",
            "objektart": input_data["objektart"],
            "wohnflaeche": f"{int(input_data['area'])} m²",
            "schlafzimmer": input_data["bedrooms"],
            "badezimmer": input_data["bathrooms"],
            "etagen": input_data["stories"],
            "parkplaetze": input_data["parking"],
            "hauptstrasse": yn_to_de(input_data["mainroad"]),
            "gaestezimmer": yn_to_de(input_data["guestroom"]),
            "keller": yn_to_de(input_data["basement"]),
            "warmwasserheizung": yn_to_de(input_data["hotwaterheating"]),
            "klimaanlage": yn_to_de(input_data["airconditioning"]),
            "bevorzugte_lage": yn_to_de(input_data["prefarea"]),
            "moeblierung": input_data["furnishingstatus"],
        },
        "kontakt": {
            "telefon": input_data["telefon"].strip(),
            "email": input_data["email"].strip(),
        },
    }

    print("\n--- JSON-Antwort ---")
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    
