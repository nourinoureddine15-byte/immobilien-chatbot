"""
╔══════════════════════════════════════════════════════════════════╗
║  Immo-Frisch KI-Bewertungsmodul                                 ║
║  Random Forest Preisschätzung + automatische Exposé-Erstellung  ║
║  Trainiert auf 800 Immobilien in NRW & Umgebung (Q1/2026)      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import os


# ------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------

def yn_to_de(value: str) -> str:
    """yes/no → Ja/Nein (für Anzeige)."""
    v = value.strip().lower()
    if v == "yes":
        return "Ja"
    if v == "no":
        return "Nein"
    return value


def normalize_yes_no(value: str) -> str:
    """ja/yes/nein/no/y/n/j → 'yes' oder 'no'."""
    v = value.strip().lower()
    if v in ("ja", "yes", "y", "j"):
        return "yes"
    if v in ("nein", "no", "n"):
        return "no"
    return value


def normalize_furnishing(value: str) -> str:
    """Möblierung normalisieren (auch deutsche Eingaben)."""
    v = value.strip().lower()
    mapping = {
        "möbliert": "furnished", "voll möbliert": "furnished",
        "furnished": "furnished",
        "teilmöbliert": "semi-furnished", "teil möbliert": "semi-furnished",
        "semi-furnished": "semi-furnished", "semi": "semi-furnished",
        "unmöbliert": "unfurnished", "un möbliert": "unfurnished",
        "unfurnished": "unfurnished", "leer": "unfurnished",
    }
    return mapping.get(v, "unfurnished")


# ------------------------------------------------------------------
# PLZ → Ort Zuordnung (aus Trainingsdaten)
# ------------------------------------------------------------------

PLZ_ORT_MAP = {}  # wird beim Training gefüllt


def plz_to_ort(plz: str) -> str:
    """Findet den Ort zur PLZ aus den Trainingsdaten."""
    plz = str(plz).strip()
    if plz in PLZ_ORT_MAP:
        return PLZ_ORT_MAP[plz]
    # Fallback: PLZ-Bereich prüfen (erste 2-3 Stellen)
    for prefix_len in (4, 3, 2):
        prefix = plz[:prefix_len]
        matches = [o for p, o in PLZ_ORT_MAP.items() if p.startswith(prefix)]
        if matches:
            return max(set(matches), key=matches.count)
    return "Bonn Zentrum"  # Default für Immo-Frisch Kerngebiet


# ------------------------------------------------------------------
# 1. Preismodell trainieren
# ------------------------------------------------------------------

FEATURE_COLS = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus", "ort",
]


def train_price_model():
    """Trainiert Random Forest auf NRW-Immobiliendaten."""

    global PLZ_ORT_MAP

    csv_path = os.path.join(os.path.dirname(__file__) or ".", "kaggle_housing.csv")
    data = pd.read_csv(csv_path)

    # PLZ → Ort Mapping aufbauen
    for _, row in data[["plz", "ort"]].drop_duplicates().iterrows():
        PLZ_ORT_MAP[str(row["plz"])] = row["ort"]

    # Zielvariable
    y = data["price"]

    # Features (ort als kategorische Variable → One-Hot-Encoding)
    X = data[FEATURE_COLS].copy()
    X = pd.get_dummies(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Performance ausgeben
    score = model.score(X_test, y_test)
    print(f"   Modell R²-Score: {score:.2%}")

    return model, X.columns


# ------------------------------------------------------------------
# 2. Preis schätzen
# ------------------------------------------------------------------

def estimate_price(model, feature_columns, input_data: dict) -> float:
    """Schätzt den Preis für eine Immobilie."""

    # Ort aus PLZ ermitteln, falls nicht direkt vorhanden
    ort = input_data.get("ort_modell")
    if not ort:
        plz = str(input_data.get("plz", ""))
        ort = plz_to_ort(plz) if plz else "Bonn Zentrum"

    row = pd.DataFrame([{
        "area":             float(input_data["area"]),
        "bedrooms":         int(input_data["bedrooms"]),
        "bathrooms":        int(input_data["bathrooms"]),
        "stories":          int(input_data["stories"]),
        "parking":          int(input_data["parking"]),
        "mainroad":         input_data["mainroad"],
        "guestroom":        input_data["guestroom"],
        "basement":         input_data["basement"],
        "hotwaterheating":  input_data["hotwaterheating"],
        "airconditioning":  input_data["airconditioning"],
        "prefarea":         input_data["prefarea"],
        "furnishingstatus": input_data["furnishingstatus"],
        "ort":              ort,
    }])

    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    price = model.predict(row)[0]
    return max(price, 0)


# ------------------------------------------------------------------
# 3. Professionelles Exposé generieren
# ------------------------------------------------------------------

def generate_description(input_data: dict, price: float) -> dict:
    """Erstellt ein professionelles Immobilien-Exposé auf Deutsch."""

    ort = input_data.get("ort", "")
    plz = input_data.get("plz", "")
    objektart = input_data.get("objektart", "Immobilie")
    area = float(input_data["area"])
    zimmer = int(input_data["bedrooms"])
    badezimmer = int(input_data["bathrooms"])
    stories = int(input_data["stories"])
    parking = int(input_data["parking"])
    furnishing = input_data["furnishingstatus"]
    mainroad = input_data["mainroad"]
    prefarea = input_data["prefarea"]
    guestroom = input_data["guestroom"]
    basement = input_data["basement"]
    hotwaterheating = input_data["hotwaterheating"]
    airconditioning = input_data["airconditioning"]

    # Ort-Anzeige mit PLZ
    ort_display = ort.strip().title() if ort else "NRW"
    if plz:
        adresse_display = f"{plz} {ort_display}"
    else:
        adresse_display = ort_display

    # ── Titel ─────────────────────────────────────────────────────
    if objektart.lower() in ("wohnung", "eigentumswohnung"):
        typ_text = "Gepflegte Eigentumswohnung"
    elif objektart.lower() in ("einfamilienhaus", "efh", "haus"):
        typ_text = "Charmantes Einfamilienhaus"
    elif objektart.lower() in ("villa",):
        typ_text = "Repräsentative Villa"
    elif objektart.lower() in ("reihenhaus",):
        typ_text = "Modernes Reihenhaus"
    elif objektart.lower() in ("doppelhaushälfte", "dhh"):
        typ_text = "Attraktive Doppelhaushälfte"
    elif objektart.lower() in ("bungalow",):
        typ_text = "Komfortabler Bungalow"
    elif objektart.lower() in ("penthouse",):
        typ_text = "Exklusives Penthouse"
    else:
        typ_text = f"Attraktive Immobilie ({objektart})"

    titel = f"{typ_text} mit {zimmer} Zimmern in {adresse_display}"

    # ── Kurzbeschreibung ──────────────────────────────────────────
    etagen_text = f"über {stories} Etagen" if stories > 1 else "auf einer Ebene"

    moebel_map = {
        "furnished":      "voll möbliert",
        "semi-furnished": "teilmöbliert",
        "unfurnished":    "unmöbliert",
    }
    moebel_text = moebel_map.get(furnishing, "unmöbliert")

    park_text = ""
    if parking == 1:
        park_text = " Ein Stellplatz ist vorhanden."
    elif parking > 1:
        park_text = f" Es stehen {parking} Stellplätze zur Verfügung."

    keller_text = " Ein Keller bietet zusätzlichen Stauraum." if basement == "yes" else ""
    klima_text = " Eine Klimaanlage sorgt für angenehmes Raumklima." if airconditioning == "yes" else ""
    gaeste_text = " Ein separates Gästezimmer rundet das Angebot ab." if guestroom == "yes" else ""

    kurzbeschreibung = (
        f"Diese gepflegte Immobilie in {adresse_display} bietet "
        f"ca. {int(area)} m² Wohnfläche {etagen_text}. "
        f"Die {zimmer} Schlafzimmer und {badezimmer} Badezimmer "
        f"sind {moebel_text}."
        f"{park_text}{keller_text}{klima_text}{gaeste_text}"
    )

    # ── Highlights ────────────────────────────────────────────────
    highlights = []
    highlights.append(f"{int(area)} m² Wohnfläche")
    highlights.append(f"{zimmer} Schlafzimmer · {badezimmer} Bad")

    if stories > 1:
        highlights.append(f"{stories} Etagen")

    if parking > 0:
        highlights.append(f"{parking} Stellplatz{'e' if parking > 1 else ''}")

    if basement == "yes":
        highlights.append("Keller vorhanden")

    if hotwaterheating == "yes":
        highlights.append("Zentralheizung")

    if airconditioning == "yes":
        highlights.append("Klimaanlage")

    if guestroom == "yes":
        highlights.append("Gästezimmer")

    if mainroad == "yes":
        highlights.append("Verkehrsgünstige Lage")
    else:
        highlights.append("Ruhige Wohnlage")

    if prefarea == "yes":
        highlights.append("Bevorzugte Wohnlage")

    # ── Lage & Markt ──────────────────────────────────────────────
    preis_pro_qm = price / area if area > 0 else 0

    if prefarea == "yes":
        lage_text = (
            f"Das Objekt befindet sich in einer begehrten Wohnlage "
            f"von {ort_display} mit hervorragender Infrastruktur, "
            f"guter Anbindung an den ÖPNV und attraktiver Nachbarschaft. "
        )
    else:
        lage_text = (
            f"Die Lage in {ort_display} überzeugt durch eine solide "
            f"Infrastruktur und gute Anbindung an das Umland. "
        )

    lage_text += (
        f"Der geschätzte Quadratmeterpreis liegt bei "
        f"ca. {round(preis_pro_qm):,} €/m².".replace(",", ".")
    )

    # ── Preis formatieren ─────────────────────────────────────────
    preis_gerundet = round(price / 1000) * 1000
    preis_formatiert = f"{preis_gerundet:,.0f} €".replace(",", ".")

    return {
        "titel":                titel,
        "kurzbeschreibung":     kurzbeschreibung,
        "highlights":           highlights,
        "lage_und_markt":       lage_text,
        "geschaetzter_preis":   preis_formatiert,
        "geschaetzter_preis_euro": round(price, 2),
        "preis_pro_qm":        round(preis_pro_qm, 2),
        "adresse":              adresse_display,
    }


# ------------------------------------------------------------------
# 4. Hauptprogramm (Standalone-Modus)
# ------------------------------------------------------------------

def main():
    print("\n🏠 Immo-Frisch KI-Bewertung wird gestartet...")
    print("   Trainiere Modell auf NRW-Immobiliendaten...")
    model, feature_columns = train_price_model()
    print("   ✅ Modell bereit.\n")

    print("─" * 50)
    print("  Bitte geben Sie die Immobiliendaten ein:")
    print("─" * 50)

    input_data = {}

    input_data["plz"]       = input("📍 Postleitzahl: ").strip()
    input_data["ort"]       = input("🏘  Ort (z.B. Bonn): ").strip()
    input_data["objektart"] = input("🏠 Objektart (Wohnung/Einfamilienhaus/...): ").strip()
    input_data["area"]      = float(input("📐 Wohnfläche in m²: "))
    input_data["bedrooms"]  = int(input("🛏  Schlafzimmer: "))
    input_data["bathrooms"] = int(input("🚿 Badezimmer: "))
    input_data["stories"]   = int(input("🏗  Etagen: "))
    input_data["parking"]   = int(input("🅿  Stellplätze: "))

    input_data["mainroad"]        = normalize_yes_no(input("🛣  An Hauptstraße? (ja/nein): "))
    input_data["guestroom"]       = normalize_yes_no(input("🛋  Gästezimmer? (ja/nein): "))
    input_data["basement"]        = normalize_yes_no(input("🏚  Keller? (ja/nein): "))
    input_data["hotwaterheating"] = normalize_yes_no(input("🔥 Zentralheizung? (ja/nein): "))
    input_data["airconditioning"] = normalize_yes_no(input("❄  Klimaanlage? (ja/nein): "))
    input_data["prefarea"]        = normalize_yes_no(input("⭐ Bevorzugte Lage? (ja/nein): "))
    input_data["furnishingstatus"]= normalize_furnishing(
        input("🪑 Möblierung (möbliert/teilmöbliert/unmöbliert): ")
    )

    # Ort für Modell aus PLZ oder Eingabe
    input_data["ort_modell"] = plz_to_ort(input_data["plz"]) if input_data["plz"] else None

    price = estimate_price(model, feature_columns, input_data)
    beschreibung = generate_description(input_data, price)

    print("\n" + "═" * 50)
    print(f"  💰 Geschätzter Preis: {beschreibung['geschaetzter_preis']}")
    print(f"  📊 Preis pro m²:     {beschreibung['preis_pro_qm']:,.0f} €/m²".replace(",", "."))
    print("═" * 50)

    print(json.dumps(beschreibung, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
