import streamlit as st
import pickle
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime

# --- Load trained model and mappings ---
# Ensure these files exist in your app directory
with open("gb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("alert_mapping.pkl", "rb") as f:
    mp = pickle.load(f)

INT_TO_COLOR = mp["INT_TO_COLOR"]  # e.g., {0: "green", 1: "yellow", 2: "orange", 3: "red"}
FEATURES = mp["FEATURES"]          # e.g., ["mag", "depth", "cdi", "mmi", "sig"]

# --- Page setup ---
st.set_page_config(page_title="Earthquake Impact Dashboard", layout="wide")

# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# --- Theme CSS (Light/Dark) with decorative touches ---
def get_theme_css(theme: str) -> str:
    if theme == "Dark":
        text_color = "white"
        card_bg = "rgba(0,0,0,0.65)"
        navbar_bg = "rgba(0,0,0,0.75)"
        hero_bg = "linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.35))"
        footer_bg = "rgba(0,0,0,0.75)"
        button_fg = "white"
        button_bg = "#333"
        button_border = "#555"
        caption_color = "white"
        shadow = "0 6px 18px rgba(0,0,0,0.35)"
    else:
        text_color = "black"
        card_bg = "white"
        navbar_bg = "rgba(255,255,255,0.9)"
        hero_bg = "linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.85))"
        footer_bg = "white"
        button_fg = "white"
        button_bg = "#1f6feb"
        button_border = "#1558c0"
        caption_color = "black"
        shadow = "0 6px 18px rgba(0,0,0,0.06)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Merriweather:wght@400&display=swap');
    html, body, .stApp {{ font-family: 'Merriweather', serif; color: {text_color} !important; }}
    h1,h2,h3,h4,h5,h6 {{ font-family: 'Poppins', sans-serif; }}

    [data-testid="stAppViewContainer"] {{
        background-image: url('https://images.unsplash.com/photo-1470770841072-f978cf4d019e?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-position: center;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .navbar {{
        display: flex; justify-content: space-between; align-items: center;
        background: {navbar_bg}; padding: 12px 24px; position: sticky; top: 0; z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        backdrop-filter: blur(3px);
    }}
    .navbar h2 {{ margin: 0; color: {text_color}; }}
    .navbar a {{ color: {text_color}; margin: 0 12px; text-decoration: none; font-weight: 600; }}
    .navbar a:hover {{ color: #FFD700; }}

    .hero {{
        padding: 70px 20px; text-align: center; color: {text_color};
        background: {hero_bg};
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }}
    .hero h1 {{ font-size: 44px; margin-bottom: 8px; animation: fadeIn 1.2s ease; }}
    .hero p  {{ font-size: 18px; opacity: 0.9; }}

    .card {{
        background: {card_bg}; color: {text_color};
        border-radius: 12px; padding: 24px; margin: 20px 0;
        box-shadow: {shadow};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .card:hover {{ transform: translateY(-5px); box-shadow: 0 12px 24px rgba(0,0,0,0.15); }}
    .card h2 {{ font-family: 'Poppins', sans-serif; }}

    .stButton button {{
        color: {button_fg} !important; background: {button_bg} !important; border: 1px solid {button_border} !important; border-radius: 8px;
        transition: background 0.3s ease;
    }}
    .stButton button:hover {{ background: #0d4aa0 !important; }}

    .prediction-box {{
        border-radius:12px; padding:18px; margin:12px 0; text-align:center; font-weight:bold; font-size:18px;
        animation: fadeIn 1s ease;
    }}
    .prediction-box.safe {{ background-color: rgba(0,255,0,0.25); }}
    .prediction-box.danger {{ background-color: rgba(255,0,0,0.25); animation: pulseRed 1s infinite; }}
    .prediction-box.moderate {{ background-color: rgba(255,165,0,0.40); animation: gentleShake 0.5s; }}

    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
    @keyframes pulseRed {{
        0% {{ box-shadow: 0 0 0px rgba(255,0,0,0.8); }}
        50% {{ box-shadow: 0 0 20px rgba(255,0,0,0.8); }}
        100% {{ box-shadow: 0 0 0px rgba(255,0,0,0.8); }}
    }}
    @keyframes gentleShake {{
        0% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-3px); }}
        50% {{ transform: translateX(3px); }}
        75% {{ transform: translateX(-3px); }}
        100% {{ transform: translateX(0); }}
    }}

    .footer {{
        background: {footer_bg}; text-align: center; padding: 12px; color: {text_color}; margin-top: 40px;
        border-top: 1px solid rgba(0,0,0,0.06);
    }}
    figcaption {{ color: {caption_color} !important; }}
    </style>
    """

# --- Login page ---
def login_page():
    st.markdown("<div class='card' style='max-width:420px;margin:100px auto;text-align:center;'><h2>🔑 Sign In</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Dashboard page ---
def dashboard_page():
    # Sidebar (theme toggle + logout)
    with st.sidebar:
        st.title("⚙️ Settings")
        theme_choice = st.selectbox("Theme", ["Light", "Dark"], index=(0 if st.session_state.theme == "Light" else 1))
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.experimental_rerun()
        st.write("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.success("Logged out successfully")
            st.experimental_rerun()

    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

    # Navbar
    st.markdown("""
    <div class="navbar">
        <h2>🌍 Earthquake Dashboard</h2>
        <div>
            <a href="#predictor">Predictor</a>
            <a href="#analytics">Analytics</a>
            <a href="#map">Map</a>
            <a href="#data-explorer">Data Explorer</a>
            <a href="#simulation">Alert Simulation</a>
            <a href="#education">Education</a>
            <a href="#info">Info</a>
            <a href="#gallery">Gallery</a>
            <a href="#about">About</a>
            <a href="#feedback">Feedback</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>Earthquake Impact Prediction</h1>
        <p>Explore seismic analytics, maps, simulations, and learn with interactive content.</p>
    </div>
    """, unsafe_allow_html=True)

    # Predictor
    st.markdown("<div id='predictor' class='card'><h2>🚨 Predictor</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        magnitude = st.number_input("Magnitude (Mw)", value=5.5, step=0.1)
        depth = st.number_input("Depth (km)", value=10.0, step=0.5)
        cdi = st.number_input("CDI (Community Internet Intensity)", value=3.0, step=0.1)
    with c2:
        mmi = st.number_input("MMI (Modified Mercalli Intensity)", value=4.0, step=0.1)
        sig = st.number_input("SIG (Significance)", value=100.0, step=1.0)

    if st.button("Predict Impact"):
        X_new = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]], columns=FEATURES)
        try:
            pred = model.predict(X_new)[0]
            alert_color = INT_TO_COLOR.get(pred, "Unknown")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            alert_color = "Unknown"

        status_map = {
            "green": "✅ Safe",
            "yellow": "⚠️ Moderate",
            "orange": "⚠️ Moderate",
            "red": "⚠️ Danger",
        }
        css_class = "safe" if str(alert_color).lower() == "green" else \
                    "danger" if str(alert_color).lower() == "red" else "moderate"

        st.markdown(f"""
        <div class='prediction-box {css_class}'>
          <h3>Prediction Result</h3>
          <p><b>Predicted Alert Color:</b> {alert_color}</p>
          <p><b>Status:</b> {status_map.get(str(alert_color).lower(), "Unknown")}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Analytics (Plotly) with fixed bar chart
    st.markdown("<div id='analytics' class='card'><h2>📊 Analytics</h2>", unsafe_allow_html=True)
    np.random.seed(42)
    df_demo = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=300, freq="D"),
        "mag": np.clip(np.random.normal(4.8, 0.8, 300), 2.5, 8.0),
        "depth": np.clip(np.random.normal(20, 15, 300), 1, 300)
    })
    a1, a2 = st.columns(2)
    with a1:
        fig_mag = px.line(df_demo, x="date", y="mag", title="Magnitude trend over time")
        fig_mag.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_mag, use_container_width=True)
    with a2:
        fig_depth = px.area(df_demo, x="date", y="depth", title="Depth variation over time")
        fig_depth.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_depth, use_container_width=True)

    hist_df = df_demo["mag"].round(1).value_counts().sort_index().reset_index()
    hist_df.columns = ["mag", "count"]  # Fix: explicit column names for Plotly
    fig_hist = px.bar(hist_df, x="mag", y="count", labels={"mag": "Magnitude", "count": "Count"},
                      title="Magnitude distribution (rounded)")
    fig_hist.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive map
    st.markdown("<div id='map' class='card'><h2>🗺️ Interactive Map</h2>", unsafe_allow_html=True)
    st.write("Sample epicenters across India. Upload real data in the Data Explorer to plot actual locations.")
    sample_map = pd.DataFrame({
        "lat": [19.0760, 28.7041, 13.0827, 22.5726, 23.0225],
        "lon": [72.8777, 77.1025, 80.2707, 88.3639, 72.5714],
        "mag": [4.5, 5.2, 6.1, 3.8, 5.9],
        "place": ["Mumbai", "Delhi", "Chennai", "Kolkata", "Ahmedabad"]
    })
    layer = pdk.Layer(
        "ScatterplotLayer",
        sample_map,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius='mag * 8000',
        pickable=True,
    )
    tooltip = {"text": "Place: {place}\nMagnitude: {mag}"}
    view_state = pdk.ViewState(latitude=21.0, longitude=78.0, zoom=4.5)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    st.markdown("</div>", unsafe_allow_html=True)

    # Historical data explorer
    st.markdown("<div id='data-explorer' class='card'><h2>📅 Historical Data Explorer</h2>", unsafe_allow_html=True)
    st.write("Upload a CSV with columns such as lat, lon, mag, depth, time, place. Filter and visualize it below.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            data.rename(columns={c: c.lower() for c in data.columns}, inplace=True)

            c3, c4, c5 = st.columns(3)
            with c3:
                mag_min = float(data.get("mag", pd.Series([0])).min()) if "mag" in data.columns else 0.0
                mag_max = float(data.get("mag", pd.Series([10])).max()) if "mag" in data.columns else 10.0
                mag_range = st.slider("Magnitude range", mag_min, mag_max, (mag_min, mag_max))
            with c4:
                depth_min = float(data.get("depth", pd.Series([0])).min()) if "depth" in data.columns else 0.0
                depth_max = float(data.get("depth", pd.Series([300])).max()) if "depth" in data.columns else 300.0
                depth_range = st.slider("Depth range (km)", depth_min, depth_max, (depth_min, depth_max))
            with c5:
                if "time" in data.columns:
                    dt = pd.to_datetime(data["time"], errors="coerce")
                    st.caption(f"Date range: {dt.min()} to {dt.max()}")

            df_f = data.copy()
            if "mag" in df_f.columns:
                df_f = df_f[(df_f["mag"] >= mag_range[0]) & (df_f["mag"] <= mag_range[1])]
            if "depth" in df_f.columns:
                df_f = df_f[(df_f["depth"] >= depth_range[0]) & (df_f["depth"] <= depth_range[1])]

            st.dataframe(df_f, use_container_width=True)
            st.caption("Filtered dataset preview")

            if {"lat", "lon"}.issubset(df_f.columns):
                layer_u = pdk.Layer(
                    "ScatterplotLayer",
                    df_f,
                    get_position='[lon, lat]',
                    get_color='[30, 120, 200, 180]',
                    get_radius='(mag if mag else 4) * 8000',
                    pickable=True,
                )
                tooltip_u = {"text": "Mag: {mag}\nDepth: {depth}\nPlace: {place}"}
                mean_lat = float(df_f["lat"].mean())
                mean_lon = float(df_f["lon"].mean())
                st.pydeck_chart(pdk.Deck(layers=[layer_u],
                                         initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=4.5),
                                         tooltip=tooltip_u))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.caption("Upload a CSV to explore historical data.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Alert simulation
    st.markdown("<div id='simulation' class='card'><h2>🔔 Alert Simulation</h2>", unsafe_allow_html=True)
    st.write("Simulate scenarios and see predicted alerts.")
    s1, s2 = st.columns(2)
    with s1:
        sim_mag = st.slider("Simulated Magnitude (Mw)", 2.0, 9.5, 6.5, 0.1)
        sim_depth = st.slider("Simulated Depth (km)", 1.0, 700.0, 15.0, 1.0)
        sim_cdi = st.slider("Simulated CDI", 1.0, 10.0, 4.0, 0.1)
    with s2:
        sim_mmi = st.slider("Simulated MMI", 1.0, 12.0, 5.0, 0.1)
        sim_sig = st.slider("Simulated SIG", 0.0, 1000.0, 250.0, 10.0)

    if st.button("Run Simulation"):
        X_sim = pd.DataFrame([[sim_mag, sim_depth, sim_cdi, sim_mmi, sim_sig]], columns=FEATURES)
        try:
            pred_sim = model.predict(X_sim)[0]
            alert_color_sim = INT_TO_COLOR.get(pred_sim, "Unknown")
        except Exception as e:
            st.error(f"Simulation error: {e}")
            alert_color_sim = "Unknown"

        css_class_sim = "safe" if str(alert_color_sim).lower() == "green" else \
                        "danger" if str(alert_color_sim).lower() == "red" else "moderate"

        st.markdown(f"""
        <div class='prediction-box {css_class_sim}'>
          <h3>Simulation Result</h3>
          <p><b>Alert Color:</b> {alert_color_sim}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Educational section
    st.markdown("<div id='education' class='card'><h2>📖 Educational Section</h2>", unsafe_allow_html=True)
    with st.expander("What do Magnitude and Intensity mean?"):
        st.write("- Magnitude (Mw) measures the energy released at the source of the earthquake.")
        st.write("- Intensity (MMI) describes the effects and shaking observed at specific locations.")
    with st.expander("Safety tips during an earthquake"):
        st.write("- Drop, cover, and hold on. Protect your head and neck.")
        st.write("- Stay away from windows, heavy furniture, and unsecured objects.")
        st.write("- If outdoors, move to an open area away from buildings and power lines.")
    with st.expander("After an earthquake"):
        st.write("- Check for injuries and hazards. Expect aftershocks.")
        st.write("- Use text messages or social media to communicate.")
        st.write("- Follow local authority instructions.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Info
    st.markdown("<div id='info' class='card'><h2>ℹ️ Earthquake Parameters Explained</h2>", unsafe_allow_html=True)
    st.write("""
    • Magnitude (Mw): Measures the size of an earthquake based on seismic wave energy.
    • Depth (km): Distance below the surface where the quake originates.
    • CDI: Crowdsourced shaking reports from communities.
    • MMI: Observed shaking intensity scale (I–XII).
    • SIG: Composite measure of an earthquake’s overall significance.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gallery
    st.markdown("<div id='gallery' class='card'><h2>📸 Earthquake Gallery</h2>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        st.image(
            "https://images.unsplash.com/photo-1641213131995-06e2cf0790d8?w=1000&auto=format&fit=crop&q=80",
            caption="Aftermath of an Earthquake",
            use_column_width=True,
        )
    with g2:
        st.image(
            "https://plus.unsplash.com/premium_photo-1716985683568-b05f58cc5c87?w=1000&auto=format&fit=crop&q=80",
            caption="Damage Caused by Earthquake",
            use_column_width=True,
        )
    g3, g4 = st.columns(2)
    with g3:
        st.image(
            "https://images.unsplash.com/photo-1508624217470-5ef0f947d8be?w=1000&auto=format&fit=crop&q=80",
            caption="Oceanic Fault Regions",
            use_column_width=True,
        )
    with g4:
        st.image(
            "https://media.istockphoto.com/id/2007470156/photo/seismic-waves-analysis.webp?a=1&b=1&s=612x612&w=0&k=20&c=zERyp5DzbFx87xfu5cq6MZDCeh9E7ua6CF-1w1pbYro=",
            caption="Seismic Waves Analysis",
            use_column_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # About
    st.markdown("<div id='about' class='card'><h2>👩‍💻 About</h2>", unsafe_allow_html=True)
    st.write("Designed by Tanushri — an artistic, educational dashboard for exploring earthquake impact.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Feedback form
    st.markdown("<div id='feedback' class='card'><h2>💬 Feedback</h2>", unsafe_allow_html=True)
    fb_name = st.text_input("Your name")
    fb_msg = st.text_area("Your feedback or suggestions")
    fb_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if st.button("Submit Feedback"):
        if fb_msg.strip():
            st.session_state.feedback.append({"name": fb_name or "Anonymous", "message": fb_msg.strip(), "time": fb_time})
            st.success("Thanks for your feedback!")
        else:
            st.warning("Please enter a message before submitting.")
    if st.session_state.feedback:
        st.write("Recent feedback:")
        st.table(pd.DataFrame(st.session_state.feedback))
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
      © 2025 Earthquake Dashboard | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


import streamlit as st

# Theme toggle in settings
theme = st.radio("Choose Theme:", ["Light", "Dark"])

# Apply theme-specific styles
if theme == "Dark":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stButton>button {
            background-color: #333333;
            color: #ffffff;
        }
        .stTextInput>div>input {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .gallery-caption {
            color: #ffffff; /* captions readable in dark mode */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton>button {
            background-color: #f0f0f0;
            color: #000000;
        }
        .stTextInput>div>input {
            background-color: #ffffff;
            color: #000000;
        }
        .gallery-caption {
            color: #000000; /* captions readable in light mode */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# --- Routing ---
def app():
    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard_page()

app()