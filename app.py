import streamlit as st
from bewertung_kaggle_mit_beschreibung import train_price_model, estimate_price, generate_description

st.set_page_config(page_title="Immobilien KI Chatbot", page_icon="🏠", layout="centered")

st.title("🏠 Immobilien KI Chatbot")
st.write("Füll das Formular aus – ich schätze den Preis und erstelle ein Exposé!")
st.divider()

@st.cache_resource
def lade_modell():
    return train_price_model()

model, feature_columns = lade_modell()

with st.form("immobilien_form"):
    st.subheader("📋 Immobiliendaten")

    col1, col2 = st.columns(2)
    with col1:
        ort = st.text_input("🏙️ Ort (z.B. Berlin)")
    with col2:
        objektart = st.text_input("🏡 Objektart (z.B. Wohnung)")

    col3, col4, col5 = st.columns(3)
    with col3:
        flaeche = st.number_input("Wohnfläche m²", min_value=10, max_value=1000, value=100)
    with col4:
        zimmer = st.number_input("Schlafzimmer", min_value=1, max_value=10, value=3)
    with col5:
        badezimmer = st.number_input("Badezimmer", min_value=1, max_value=5, value=2)

    col6, col7 = st.columns(2)
    with col6:
        etagen = st.number_input("Etagen", min_value=1, max_value=5, value=2)
    with col7:
        parkplaetze = st.number_input("Parkplätze", min_value=0, max_value=5, value=1)

    st.subheader("⚙️ Ausstattung")
    col8, col9, col10 = st.columns(3)
    with col8:
        hauptstrasse = st.selectbox("Hauptstraße?", ["yes", "no"])
        gaestezimmer = st.selectbox("Gästezimmer?", ["yes", "no"])
    with col9:
        keller = st.selectbox("Keller?", ["yes", "no"])
        heizung = st.selectbox("Warmwasser?", ["yes", "no"])
    with col10:
        klima = st.selectbox("Klimaanlage?", ["yes", "no"])
        bevorzugte_lage = st.selectbox("Top-Lage?", ["yes", "no"])

    moeblierung = st.selectbox("🛋️ Möblierung", ["furnished", "semi-furnished", "unfurnished"])

    abschicken = st.form_submit_button("🔍 Preis schätzen", use_container_width=True)

if abschicken:
    with st.spinner("KI berechnet den Preis..."):
        input_data = {
            "ort": ort, "objektart": objektart,
            "area": flaeche, "bedrooms": zimmer,
            "bathrooms": badezimmer, "stories": etagen,
            "parking": parkplaetze, "mainroad": hauptstrasse,
            "guestroom": gaestezimmer, "basement": keller,
            "hotwaterheating": heizung, "airconditioning": klima,
            "prefarea": bevorzugte_lage, "furnishingstatus": moeblierung,
        }
        preis = estimate_price(model, feature_columns, input_data)
        beschreibung = generate_description(input_data, preis)

    st.divider()
    st.metric(label="💰 Geschätzter Marktwert", value=beschreibung['geschaetzter_preis'])

    st.divider()
    st.subheader("📄 Automatisch generiertes Exposé")
    st.markdown(f"### {beschreibung['titel']}")
    st.write(beschreibung["kurzbeschreibung"])
    st.write(beschreibung["lage_und_markt"])

    st.subheader("✅ Highlights")
    col_a, col_b = st.columns(2)
    highlights = beschreibung["highlights"]
    for i, h in enumerate(highlights):
        if i % 2 == 0:
            col_a.success(h)
        else:
            col_b.success(h)