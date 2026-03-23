import streamlit as st
from bewertung_kaggle_mit_beschreibung import train_price_model, estimate_price, generate_description

st.set_page_config(
    page_title="Immo-Frisch | KI Preisschaetzer",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
<style>
.header-bar {
    background: #1a3a5c;
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title { color: white; font-size: 20px; font-weight: 600; margin: 0; }
.header-sub { color: #93c5fd; font-size: 13px; margin: 2px 0 0 0; }
.status-badge {
    background: rgba(34,197,94,0.2);
    border: 1px solid #22c55e;
    color: #86efac;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.info-banner {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 20px;
    font-size: 13px;
    color: #1e40af;
}
.result-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    margin-top: 8px;
}
.price-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.price-value {
    font-size: 36px;
    font-weight: 700;
    color: #1a3a5c;
    margin: 0 0 4px 0;
}
.price-note { font-size: 12px; color: #94a3b8; margin-bottom: 20px; }
.expose-title { font-size: 15px; font-weight: 600; color: #1a3a5c; margin-bottom: 8px; }
.expose-text { font-size: 13px; color: #475569; line-height: 1.7; margin-bottom: 12px; }
.highlight-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.highlight-item {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #0369a1;
    font-weight: 500;
}
.footer-bar {
    background: #1a3a5c;
    padding: 14px 20px;
    border-radius: 10px;
    margin-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-text { color: #93c5fd; font-size: 12px; margin: 0; }
.footer-link { color: white; font-size: 12px; font-weight: 500; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <div>
        <p class="header-title">🏠 Immo-Frisch KI Preisschaetzer</p>
        <p class="header-sub">Ihr Immobilienexperte im Rheinland &middot; Bonn &amp; Umgebung</p>
    </div>
    <div class="status-badge">&#9679; Online</div>
</div>
<div class="info-banner">
    Willkommen! Unser KI-System schaetzt den Marktwert Ihrer Immobilie in Sekunden - kostenlos und unverbindlich.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def lade_modell():
    return train_price_model()

model, feature_columns = lade_modell()

fragen = [
    ("ort",             "In welcher Stadt liegt die Immobilie? (z.B. Bonn, Koeln, Remagen)"),
    ("objektart",       "Um welche Art von Immobilie handelt es sich? (z.B. Wohnung, Einfamilienhaus)"),
    ("area",            "Wie gross ist die Wohnflaeche in m2? (z.B. 120)"),
    ("bedrooms",        "Wie viele Schlafzimmer hat die Immobilie?"),
    ("bathrooms",       "Wie viele Badezimmer gibt es?"),
    ("stories",         "Ueber wie viele Etagen erstreckt sich die Immobilie?"),
    ("parking",         "Wie viele Stellplaetze gibt es?"),
    ("mainroad",        "Liegt die Immobilie an einer Hauptstrasse? (ja / nein)"),
    ("guestroom",       "Gibt es ein separates Gaestezimmer? (ja / nein)"),
    ("basement",        "Verfuegt die Immobilie ueber einen Keller? (ja / nein)"),
    ("hotwaterheating", "Gibt es eine Warmwasserheizung? (ja / nein)"),
    ("airconditioning", "Ist eine Klimaanlage vorhanden? (ja / nein)"),
    ("prefarea",        "Befindet sich die Immobilie in einer bevorzugten Wohnlage? (ja / nein)"),
    ("furnishingstatus","Wie ist der Moeblierungsstatus? (furnished / semi-furnished / unfurnished)"),
]

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.antworten = {}
    st.session_state.schritt = 0
    st.session_state.fertig = False
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Guten Tag! Ich bin der KI-Assistent von **Immo-Frisch**. Ich begleite Sie Schritt fuer Schritt durch die Bewertung Ihrer Immobilie.\n\n**Frage 1 von 14:** " + fragen[0][1],
        "type": "text"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "result":
            highlights_html = "".join([
                f'<div class="highlight-item">&#10003; {h}</div>'
                for h in msg["highlights"]
            ])
            st.markdown(f"""
<div class="result-box">
    <div class="price-label">Geschaetzter Marktwert</div>
    <div class="price-value">{msg['preis']}</div>
    <div class="price-note">Basierend auf aktuellen Marktdaten &middot; KI-Schaetzung</div>
    <hr style="border:none;border-top:1px solid #f1f5f9;margin:16px 0;">
    <div class="expose-title">📄 {msg['titel']}</div>
    <div class="expose-text">{msg['beschreibung']}</div>
    <div class="expose-text">{msg['lage']}</div>
    <div class="price-label" style="margin-top:12px;">Highlights</div>
    <div class="highlight-grid">{highlights_html}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("📞 **Interesse an einer professionellen Beratung?** Kontaktieren Sie uns unter [immo-frisch.de](https://immo-frisch.de/immo-frisch/kontakt/)")
        else:
            st.markdown(msg["content"])

if not st.session_state.fertig:
    if antwort := st.chat_input("Ihre Antwort..."):
        st.session_state.messages.append({"role": "user", "content": antwort, "type": "text"})
        key, _ = fragen[st.session_state.schritt]
        st.session_state.antworten[key] = antwort
        st.session_state.schritt += 1

        if st.session_state.schritt < len(fragen):
            naechste = fragen[st.session_state.schritt]
            bestaetigung = f"Vielen Dank! ✅\n\n**Frage {st.session_state.schritt + 1} von {len(fragen)}:** {naechste[1]}"
            st.session_state.messages.append({"role": "assistant", "content": bestaetigung, "type": "text"})
        else:
            st.session_state.fertig = True
            a = st.session_state.antworten

            def ja_nein(v):
                return "yes" if v.strip().lower() in ["ja","j","yes","y"] else "no"

            input_data = {
                "ort": a["ort"], "objektart": a["objektart"],
                "area": float(a["area"]), "bedrooms": int(a["bedrooms"]),
                "bathrooms": int(a["bathrooms"]), "stories": int(a["stories"]),
                "parking": int(a["parking"]),
                "mainroad": ja_nein(a["mainroad"]),
                "guestroom": ja_nein(a["guestroom"]),
                "basement": ja_nein(a["basement"]),
                "hotwaterheating": ja_nein(a["hotwaterheating"]),
                "airconditioning": ja_nein(a["airconditioning"]),
                "prefarea": ja_nein(a["prefarea"]),
                "furnishingstatus": a["furnishingstatus"].strip().lower(),
            }

            preis = estimate_price(model, feature_columns, input_data)
            beschreibung = generate_description(input_data, preis)

            st.session_state.messages.append({
                "role": "assistant",
                "type": "result",
                "preis": beschreibung["geschaetzter_preis"],
                "titel": beschreibung["titel"],
                "beschreibung": beschreibung["kurzbeschreibung"],
                "lage": beschreibung["lage_und_markt"],
                "highlights": beschreibung["highlights"],
            })

        st.rerun()

st.markdown("""
<div class="footer-bar">
    <p class="footer-text">&#169; 2025 Immo-Frisch &middot; Ihr Immobilienexperte im Rheinland</p>
    <a class="footer-link" href="https://immo-frisch.de" target="_blank">immo-frisch.de &#8594;</a>
</div>
""", unsafe_allow_html=True)
