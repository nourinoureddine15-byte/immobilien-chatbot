"""
╔══════════════════════════════════════════════════════╗
║  Immo-Frisch KI Chatbot — Streamlit App             ║
║  Immobilienbewertung per Dialog                      ║
║  Für: immo-frisch.de · Bonn & Rheinland             ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
from bewertung_kaggle_mit_beschreibung import (
    train_price_model, estimate_price, generate_description,
    normalize_yes_no, normalize_furnishing, plz_to_ort,
)

# ── Seiten-Konfiguration ─────────────────────────────────────────
st.set_page_config(
    page_title="Immo-Frisch | KI Immobilienbewertung",
    page_icon="🏠",
    layout="centered",
)

# ── CSS Design ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.header-bar {
    background: linear-gradient(135deg, #1a3a5c 0%, #1e4d7b 100%);
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title {
    color: white;
    font-size: 21px;
    font-weight: 700;
    margin: 0;
    font-family: 'Inter', sans-serif;
}
.header-sub {
    color: #93c5fd;
    font-size: 13px;
    margin: 3px 0 0 0;
    font-family: 'Inter', sans-serif;
}
.status-badge {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #86efac;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}
.info-banner {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 12px 18px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 18px;
    font-size: 13.5px;
    color: #1e40af;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}
.progress-bar {
    background: #e2e8f0;
    border-radius: 6px;
    height: 6px;
    margin: 0 0 20px 0;
    overflow: hidden;
}
.progress-fill {
    background: linear-gradient(90deg, #2563eb, #1a3a5c);
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.result-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin-top: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.price-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
    font-family: 'Inter', sans-serif;
}
.price-value {
    font-size: 38px;
    font-weight: 700;
    color: #1a3a5c;
    margin: 0 0 2px 0;
    font-family: 'Inter', sans-serif;
}
.price-sub {
    font-size: 14px;
    font-weight: 500;
    color: #2563eb;
    margin: 0 0 4px 0;
    font-family: 'Inter', sans-serif;
}
.price-note {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
}
.expose-title {
    font-size: 16px;
    font-weight: 600;
    color: #1a3a5c;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
}
.expose-text {
    font-size: 13.5px;
    color: #475569;
    line-height: 1.75;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
}
.highlight-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}
.highlight-item {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 9px 13px;
    font-size: 12.5px;
    color: #0369a1;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}
.footer-bar {
    background: linear-gradient(135deg, #1a3a5c 0%, #1e4d7b 100%);
    padding: 16px 22px;
    border-radius: 12px;
    margin-top: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-text {
    color: #93c5fd;
    font-size: 12px;
    margin: 0;
    font-family: 'Inter', sans-serif;
}
.footer-link {
    color: white;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    font-family: 'Inter', sans-serif;
}
.divider {
    border: none;
    border-top: 1px solid #f1f5f9;
    margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div>
        <p class="header-title">🏠 Immo-Frisch KI Immobilienbewertung</p>
        <p class="header-sub">Ihr Immobilienexperte im Rheinland &middot; Bonn &amp; Umgebung</p>
    </div>
    <div class="status-badge">&#9679; Online</div>
</div>
<div class="info-banner">
    Willkommen! Unser KI-System bewertet Ihre Immobilie in NRW anhand aktueller
    Marktdaten (Q1/2026). Beantworten Sie 15 kurze Fragen &ndash; kostenlos und unverbindlich.
</div>
""", unsafe_allow_html=True)

# ── Modell laden (cached) ────────────────────────────────────────
@st.cache_resource
def lade_modell():
    return train_price_model()

model, feature_columns = lade_modell()

# ── Fragen-Katalog ───────────────────────────────────────────────
FRAGEN = [
    ("plz",
     "Wie lautet die **Postleitzahl** der Immobilie? (z.B. 53111)"),
    ("ort",
     "In welcher **Stadt / Stadtteil** liegt die Immobilie? (z.B. Bonn, Köln Ehrenfeld)"),
    ("objektart",
     "Um welche **Objektart** handelt es sich?\n\n"
     "_(z.B. Einfamilienhaus, Wohnung, Villa, Reihenhaus, Doppelhaushälfte, Bungalow, Penthouse)_"),
    ("area",
     "Wie groß ist die **Wohnfläche** in m²? (z.B. 120)"),
    ("bedrooms",
     "Wie viele **Schlafzimmer** hat die Immobilie?"),
    ("bathrooms",
     "Wie viele **Badezimmer** gibt es?"),
    ("stories",
     "Über wie viele **Etagen** erstreckt sich die Immobilie?"),
    ("parking",
     "Wie viele **Stellplätze** gibt es? (0 wenn keine)"),
    ("mainroad",
     "Liegt die Immobilie an einer **Hauptstraße**? _(ja / nein)_"),
    ("guestroom",
     "Gibt es ein separates **Gästezimmer**? _(ja / nein)_"),
    ("basement",
     "Verfügt die Immobilie über einen **Keller**? _(ja / nein)_"),
    ("hotwaterheating",
     "Gibt es eine **Zentralheizung** mit Warmwasser? _(ja / nein)_"),
    ("airconditioning",
     "Ist eine **Klimaanlage** vorhanden? _(ja / nein)_"),
    ("prefarea",
     "Befindet sich die Immobilie in einer **bevorzugten Wohnlage**? _(ja / nein)_"),
    ("furnishingstatus",
     "Wie ist der **Möblierungsstatus**?\n\n"
     "_(möbliert / teilmöbliert / unmöbliert)_"),
]

TOTAL_FRAGEN = len(FRAGEN)

# ── Session State initialisieren ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.antworten = {}
    st.session_state.schritt = 0
    st.session_state.fertig = False

    # Begrüßung + erste Frage
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Guten Tag! Ich bin der KI-Assistent von **Immo-Frisch**. "
            "Ich begleite Sie Schritt für Schritt durch die Bewertung "
            "Ihrer Immobilie.\n\n"
            f"**Frage 1 von {TOTAL_FRAGEN}:** {FRAGEN[0][1]}"
        ),
        "type": "text",
    })

# ── Fortschrittsbalken ───────────────────────────────────────────
if not st.session_state.fertig:
    progress = (st.session_state.schritt / TOTAL_FRAGEN) * 100
    st.markdown(
        f'<div class="progress-bar">'
        f'<div class="progress-fill" style="width:{progress}%"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Chat-Verlauf anzeigen ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "result":
            highlights_html = "".join(
                f'<div class="highlight-item">&#10003; {h}</div>'
                for h in msg["highlights"]
            )
            st.markdown(f"""
<div class="result-box">
    <div class="price-label">Geschätzter Marktwert</div>
    <div class="price-value">{msg['preis']}</div>
    <div class="price-sub">{msg['preis_qm']}</div>
    <div class="price-note">Basierend auf aktuellen NRW-Marktdaten (Q1/2026) &middot; KI-Schätzung</div>
    <hr class="divider">
    <div class="expose-title">📄 {msg['titel']}</div>
    <div class="expose-text">{msg['beschreibung']}</div>
    <div class="expose-text">{msg['lage']}</div>
    <div class="price-label" style="margin-top:14px;">Highlights</div>
    <div class="highlight-grid">{highlights_html}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(
                "📞 **Interesse an einer professionellen Beratung?** "
                "Kontaktieren Sie uns unter "
                "[immo-frisch.de](https://immo-frisch.de/immo-frisch/kontakt/)"
            )
        else:
            st.markdown(msg["content"])

# ── Chat-Input verarbeiten ────────────────────────────────────────
if not st.session_state.fertig:
    if antwort := st.chat_input("Ihre Antwort..."):

        # Antwort speichern
        st.session_state.messages.append({
            "role": "user", "content": antwort, "type": "text",
        })

        key, _ = FRAGEN[st.session_state.schritt]
        st.session_state.antworten[key] = antwort.strip()
        st.session_state.schritt += 1

        # ── Nächste Frage oder Ergebnis? ──────────────────────────
        if st.session_state.schritt < TOTAL_FRAGEN:
            naechste_key, naechste_frage = FRAGEN[st.session_state.schritt]
            bestaetigung = (
                f"Vielen Dank! ✅\n\n"
                f"**Frage {st.session_state.schritt + 1} von {TOTAL_FRAGEN}:** "
                f"{naechste_frage}"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": bestaetigung,
                "type": "text",
            })
        else:
            # ── Alle Fragen beantwortet → Preis berechnen ────────
            st.session_state.fertig = True
            a = st.session_state.antworten

            # Eingaben normalisieren
            plz_input = a.get("plz", "").strip()
            ort_input = a.get("ort", "").strip()

            # Ort für das Modell bestimmen (aus PLZ oder Eingabe)
            ort_modell = plz_to_ort(plz_input) if plz_input else None

            input_data = {
                "plz":              plz_input,
                "ort":              ort_input,
                "ort_modell":       ort_modell,
                "objektart":        a.get("objektart", "Immobilie"),
                "area":             float(a.get("area", 100)),
                "bedrooms":         int(a.get("bedrooms", 3)),
                "bathrooms":        int(a.get("bathrooms", 1)),
                "stories":          int(a.get("stories", 1)),
                "parking":          int(a.get("parking", 0)),
                "mainroad":         normalize_yes_no(a.get("mainroad", "nein")),
                "guestroom":        normalize_yes_no(a.get("guestroom", "nein")),
                "basement":         normalize_yes_no(a.get("basement", "nein")),
                "hotwaterheating":  normalize_yes_no(a.get("hotwaterheating", "ja")),
                "airconditioning":  normalize_yes_no(a.get("airconditioning", "nein")),
                "prefarea":         normalize_yes_no(a.get("prefarea", "nein")),
                "furnishingstatus": normalize_furnishing(a.get("furnishingstatus", "unmöbliert")),
            }

            # Preis schätzen
            preis = estimate_price(model, feature_columns, input_data)

            # Exposé erstellen
            beschreibung = generate_description(input_data, preis)

            # Preis pro m² formatieren
            preis_qm = beschreibung.get("preis_pro_qm", 0)
            preis_qm_str = f"ca. {round(preis_qm):,} €/m²".replace(",", ".")

            # Lade-Nachricht
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Vielen Dank für Ihre Angaben! ✅\n\n⏳ **Bewertung wird berechnet...**",
                "type": "text",
            })

            # Ergebnis-Nachricht
            st.session_state.messages.append({
                "role":         "assistant",
                "type":         "result",
                "preis":        beschreibung["geschaetzter_preis"],
                "preis_qm":     preis_qm_str,
                "titel":        beschreibung["titel"],
                "beschreibung": beschreibung["kurzbeschreibung"],
                "lage":         beschreibung["lage_und_markt"],
                "highlights":   beschreibung["highlights"],
            })

        st.rerun()

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <p class="footer-text">&#169; 2025 Immo-Frisch &middot; Ihr Immobilienexperte im Rheinland</p>
    <a class="footer-link" href="https://immo-frisch.de" target="_blank">immo-frisch.de &#8594;</a>
</div>
""", unsafe_allow_html=True)
