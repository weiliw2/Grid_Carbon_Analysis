"""
Grid Carbon-Intensity Emulator - Streamlit Dashboard
Theme: Sage & Stone (Large Text + Base Map Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add src to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# -----------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# -----------------------------------------------------------------------------

def setup_data():
    """Setup data files if they don't exist"""
    required_files = [
        'data/processed/country_carbon_intensity.csv',
        'data/processed/carbon_emulator_model.pkl',
        'data/processed/ml_features.csv'
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        st.info("🔄 First-time setup: Generating data files... This takes 2-3 minutes.")
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        progress_bar = st.progress(0)
        
        try:
            import phase1_data_fetch as p1
            df = p1.fetch_power_plant_data()
            if df is not None: p1.explore_data(df)
            progress_bar.progress(33)
            
            import phase2_carbon_intensity as p2
            df = p2.load_power_plant_data()
            df = p2.map_emission_factors(df)
            df = p2.calculate_plant_emissions(df)
            country_data = p2.calculate_country_carbon_intensity(df)
            
            df.to_csv('data/processed/plants_with_emissions.csv', index=False)
            country_data.to_csv('data/processed/country_carbon_intensity.csv', index=False)
            progress_bar.progress(66)
            
            import phase3_ml_emulator as p3
            plants_df, country_df = p2.load_power_plant_data(), country_data
            fuel_features = p3.create_fuel_mix_features(plants_df)
            X, y, ml_data = p3.prepare_ml_dataset(fuel_features, country_df)
            
            model_result = p3.train_models(X, y)
            if model_result[0] is not None:
                joblib.dump(model_result[0], 'data/processed/carbon_emulator_model.pkl')
                X.to_csv('data/processed/ml_features.csv')
                y.to_csv('data/processed/ml_targets.csv')
            
            progress_bar.progress(100)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during setup: {str(e)}")
            st.stop()

setup_data()

st.set_page_config(
    page_title="Grid Carbon Emulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. DESIGN SYSTEM (CSS) - UPDATED FOR LARGER TEXT
# -----------------------------------------------------------------------------

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* --- COLOR VARIABLES --- */
    :root {
        --sage-dark: #354F52;
        --sage-med: #52796F;
        --sage-light: #84A98C;
        --text-main: #2F3E46;
        --text-sub: #5F6F65;
        --bg-main: #F9F9F9;
        --card-bg: #FFFFFF;
        --border-color: #E0E7E4;
    }
    
    /* --- GLOBAL TYPOGRAPHY --- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
        background-color: var(--bg-main);
    }
    
    /* Page Title: Increased to ~42px */
    h1 {
        font-size: 2.75rem !important; 
        font-weight: 700 !important;
        color: var(--sage-dark) !important;
        margin-bottom: 0.75rem !important;
        padding-top: 0 !important;
        letter-spacing: -0.02em;
    }
    
    /* Section Headers: Increased to ~26px */
    h2, h3 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em;
    }
    
    /* Body Text: Increased to ~18px */
    p, li, .stMarkdown {
        font-size: 1.125rem !important; /* 18px */
        line-height: 1.6 !important;
        color: var(--text-main) !important;
    }
    
    /* Labels / Captions: Increased to ~15px */
    .caption, [data-testid="stMetricLabel"] {
        font-size: 1rem !important; /* 16px */
        color: var(--text-sub) !important;
        font-weight: 500 !important;
    }
    
    /* --- LAYOUT & CONTAINERS --- */
    .block-container {
        padding: 3rem 4rem;
        max-width: 1400px; /* Slightly wider to accommodate larger text */
    }
    
    .content-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics - Larger Impact */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important; /* 36px */
        font-weight: 700 !important;
        color: var(--sage-dark) !important;
    }
    
    /* --- COMPONENT STYLING --- */
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.125rem; /* 18px */
        font-weight: 600;
        color: var(--text-sub);
        padding: 1rem 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--sage-med) !important;
        border-bottom: 3px solid var(--sage-med);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--sage-med);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1.125rem; /* 18px */
    }
    
    .stButton > button:hover {
        background-color: var(--sage-dark);
        color: white;
    }
    
    /* Inputs */
    .stTextInput > div > div, .stSelectbox > div > div, .stNumberInput > div > div {
        font-size: 1.125rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS (THEME UPDATE)
# -----------------------------------------------------------------------------

def apply_chart_theme(fig):
    """Applies the minimal Sage/Climate-Tech theme to Plotly figures"""
    fig.update_layout(
        font={'family': "Inter", 'color': "#2F3E46", 'size': 14}, # Base font size for charts
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor="#E0E7E4"),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F2", zeroline=False),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter"
        )
    )
    return fig

@st.cache_data
def load_data():
    country_data = pd.read_csv('data/processed/country_carbon_intensity.csv')
    ml_features = pd.read_csv('data/processed/ml_features.csv', index_col=0)
    return country_data, ml_features

@st.cache_resource
def load_model():
    return joblib.load('data/processed/carbon_emulator_model.pkl')

# -----------------------------------------------------------------------------
# 4. MAIN APP LOGIC
# -----------------------------------------------------------------------------

try:
    country_data, ml_features = load_data()
    model = load_model()
    data_loaded = True
except Exception:
    data_loaded = False

if data_loaded:
    # --- HERO SECTION ---
    st.markdown("<h1>Grid Carbon Emulator</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #5F6F65; font-size: 1.25rem; margin-bottom: 2.5rem; max-width: 900px;">
        Explore grid carbon intensity across 167 countries. 
        Simulate energy transitions and evaluate infrastructure decisions using machine learning.
    </p>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Global Overview", "Policy Simulator", "Country Analysis", "Data Center Calc"])
    
    # === TAB 1: GLOBAL OVERVIEW ===
    with tab1:
        # Top-level Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Global Average", f"{country_data['carbon_intensity_gco2_kwh'].mean():.0f} gCO₂/kWh")
        with col2:
            cleanest = country_data.nsmallest(1, 'carbon_intensity_gco2_kwh').iloc[0]
            st.metric("Lowest Intensity", f"{cleanest['country']}")
        with col3:
            dirtiest = country_data.nlargest(1, 'carbon_intensity_gco2_kwh').iloc[0]
            st.metric("Highest Intensity", f"{dirtiest['country']}")
        with col4:
            st.metric("Avg Renewable Share", f"{country_data['renewable_percentage'].mean():.1f}%")
        
        st.markdown("---")
        
        # --- UPDATED MAP SECTION ---
        st.markdown("### Carbon Intensity by Region")
        
        # Colorscale: Sage (Low Carbon) -> Slate (High Carbon)
        custom_scale = [
            [0.0, '#84A98C'],  # Light Sage (Clean)
            [0.2, '#6B9080'],
            [0.5, '#A4C3B2'],  # Mid
            [0.7, '#6C757D'],  # Greyish
            [1.0, '#2F3E46']   # Dark Slate (Dirty)
        ]

        fig = px.choropleth(
            country_data,
            locations='country',
            locationmode='ISO-3',
            color='carbon_intensity_gco2_kwh',
            hover_name='country',
            hover_data={
                'carbon_intensity_gco2_kwh': ':.0f',
                'renewable_percentage': ':.1f',
                'country': False
            },
            color_continuous_scale=px.colors.sequential.Tealgrn,
            labels={'carbon_intensity_gco2_kwh': 'gCO₂/kWh'}
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=0.5
        )
        
        # --- MAP VISUAL UPGRADE ---
        # Added showland, showcountries, and specific colors for the "Base Map" feel
        fig.update_layout(
            height=600, # Taller for visibility
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="#FFFFFF",
                showland=True,
                landcolor="#F0F4F2", # Soft light grey land (Base layer)
                showcountries=True,
                countrycolor="#FFFFFF", # White borders
                projection_type='natural earth', # Better looking projection
                bgcolor='rgba(0,0,0,0)',
                lakecolor='rgba(0,0,0,0)'
            ),
            coloraxis_colorbar=dict(
                title="Intensity (gCO₂)",
                thickness=15,
                len=0.6,
                yanchor="middle", y=0.5,
                tickfont=dict(color="#5F6F65", size=14)
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top/Bottom Tables
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cleanest Grids**")
            cleanest_10 = country_data.nsmallest(5, 'carbon_intensity_gco2_kwh')[
                ['country', 'carbon_intensity_gco2_kwh', 'renewable_percentage']
            ]
            st.dataframe(
                cleanest_10, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "country": "Country",
                    "carbon_intensity_gco2_kwh": st.column_config.NumberColumn("Intensity (gCO₂)", format="%d"),
                    "renewable_percentage": st.column_config.ProgressColumn("Renewable %", format="%.1f%%", min_value=0, max_value=100)
                }
            )
            
        with col2:
            st.markdown("**Highest Intensity Grids**")
            dirtiest_10 = country_data.nlargest(5, 'carbon_intensity_gco2_kwh')[
                ['country', 'carbon_intensity_gco2_kwh', 'renewable_percentage']
            ]
            st.dataframe(
                dirtiest_10, 
                hide_index=True, 
                use_container_width=True,
                column_config={
                    "country": "Country",
                    "carbon_intensity_gco2_kwh": st.column_config.NumberColumn("Intensity (gCO₂)", format="%d"),
                    "renewable_percentage": st.column_config.ProgressColumn("Renewable %", format="%.1f%%", min_value=0, max_value=100)
                }
            )

    # === TAB 2: POLICY SIMULATOR ===
    with tab2:
        st.markdown("### Energy Mix Transition Simulator")
        st.markdown("Model the carbon reduction impact of replacing coal capacity with solar.")
        
        countries_with_coal = ml_features[ml_features.get('Coal_pct', 0) > 5].index.tolist()
        
        if countries_with_coal:
            col_sel, col_blank = st.columns([1, 2])
            with col_sel:
                selected_country = st.selectbox("Select Country", countries_with_coal, index=0)
            
            baseline_features = ml_features.loc[[selected_country]].copy()
            baseline_intensity = model.predict(baseline_features)[0]
            coal_pct = baseline_features.get('Coal_pct', pd.Series([0])).values[0]

            st.markdown("---")
            
            # Layout: Controls on Left, Results on Right
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.markdown('<div class="content-card">', unsafe_allow_html=True)
                st.markdown("#### Adjust Mix")
                coal_reduction = st.slider(
                    "Coal → Solar Conversion (%)",
                    0, min(100, int(coal_pct)), 0, 5
                )
                
                # Calculation Logic
                scenario_features = baseline_features.copy()
                if coal_reduction > 0:
                    scenario_features['Coal_pct'] -= coal_reduction
                    scenario_features['Solar_pct'] += coal_reduction
                    # Recalc ratios
                    renewables = ['Solar', 'Wind', 'Hydro', 'Geothermal']
                    ren_cols = [f'{r}_pct' for r in renewables if f'{r}_pct' in scenario_features.columns]
                    scenario_features['renewable_ratio'] = scenario_features[ren_cols].sum(axis=1).values[0] / 100
                
                new_intensity = model.predict(scenario_features)[0]
                reduction_pct = ((baseline_intensity - new_intensity) / baseline_intensity) * 100

                st.markdown("#### Impact Summary")
                st.write(f"Before: **{baseline_intensity:.0f}** gCO₂/kWh")
                st.write(f"After: **{new_intensity:.0f}** gCO₂/kWh")
                st.markdown(f"**Reduction: {reduction_pct:.1f}%**")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                # Comparison Chart
                chart_data = pd.DataFrame({
                    'Scenario': ['Current Mix', 'Projected Mix'],
                    'Intensity': [baseline_intensity, new_intensity],
                    'Color': ['#5F6F65', '#52796F'] # Grey vs Sage
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_data['Scenario'],
                    y=chart_data['Intensity'],
                    marker_color=chart_data['Color'],
                    width=0.4
                ))
                
                fig = apply_chart_theme(fig)
                fig.update_layout(title="Carbon Intensity Forecast", height=400)
                st.plotly_chart(fig, use_container_width=True)

    # === TAB 3: COUNTRY ANALYSIS ===
    with tab3:
        all_countries = sorted(country_data['country'].tolist())
        sel_country = st.selectbox("Search Country", all_countries, index=all_countries.index('USA') if 'USA' in all_countries else 0)
        
        c_info = country_data[country_data['country'] == sel_country].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Grid Intensity", f"{c_info['carbon_intensity_gco2_kwh']:.0f}")
        with col2:
            st.metric("Renewable Share", f"{c_info['renewable_percentage']:.1f}%")
        with col3:
            st.metric("Dominant Source", c_info['dominant_fuel'])
            
        # Fuel Mix Donut
        if sel_country in ml_features.index:
            st.markdown("### Generation Mix")
            fuel_cols = [c for c in ml_features.columns if c.endswith('_pct')]
            f_data = ml_features.loc[sel_country, fuel_cols]
            f_data = f_data[f_data > 0].sort_values(ascending=False)
            
            # Subtle Palette for Donut
            sage_palette = ['#2F3E46', '#354F52', '#52796F', '#84A98C', '#CAD2C5', '#E0E7E4']
            
            fig = px.pie(
                values=f_data.values,
                names=[c.replace('_pct', '') for c in f_data.index],
                hole=0.6,
                color_discrete_sequence=sage_palette
            )
            fig.update_traces(textposition='outside', textinfo='percent+label')
            fig = apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 4: DATA CENTER CALCULATOR ===
    with tab4:
        st.markdown("### Data Center Carbon Cost")
        st.markdown("Estimate carbon tax liability based on grid location.")
        
        # Input Section
        with st.container():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                dc_mw = st.number_input("Capacity (MW)", 1, 500, 50)
            with c2:
                dc_loc = st.selectbox("Location", sorted(country_data['country'].tolist()))
            with c3:
                tax = st.number_input("Carbon Tax ($/t)", 0, 200, 50)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Calc
        loc_data = country_data[country_data['country'] == dc_loc].iloc[0]
        intensity = loc_data['carbon_intensity_gco2_kwh']
        emissions = (dc_mw * 0.7 * 1.5 * 8760 * intensity) / 1000 # tonnes roughly
        cost = emissions * tax
        
        st.markdown("#### Annual Projection")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Emissions", f"{emissions:,.0f} tonnes")
        with m2: st.metric("Est. Tax Cost", f"${cost:,.0f}")
        with m3: st.metric("Grid Intensity", f"{intensity:.0f} gCO₂/kWh")
        
        if intensity > 400:
            st.warning(f"⚠️ **High Carbon Risk:** {dc_loc} has a carbon-intensive grid. Consider onsite renewables.")
        elif intensity < 100:
            st.success(f"✅ **Low Carbon:** {dc_loc} is an optimal location for sustainable operations.")
        else:
            st.info(f"ℹ️ **Moderate:** {dc_loc} has average intensity. Efficiency improvements recommended.")

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #5F6F65; font-size: 1rem;">
            Grid Carbon Emulator • Built with Streamlit • Data Source: WRI
        </div>
        """, 
        unsafe_allow_html=True
    )

else:
    st.warning("Data pipeline not run.")