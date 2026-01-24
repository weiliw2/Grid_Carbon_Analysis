"""
Grid Carbon-Intensity Emulator - Streamlit Dashboard
Interactive web application to explore carbon intensity predictions
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

# Auto-setup: Check if data exists, if not run the pipeline
def setup_data():
    """Setup data files if they don't exist"""
    required_files = [
        'data/processed/country_carbon_intensity.csv',
        'data/processed/carbon_emulator_model.pkl',
        'data/processed/ml_features.csv'
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        st.info("First-time setup: Generating data files... This takes 2-3 minutes.")
        
        # Create directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import phase modules
            status_text.text("Phase 1/3: Fetching global power plant data...")
            progress_bar.progress(10)
            
            import phase1_data_fetch as p1
            df = p1.fetch_power_plant_data()
            if df is not None:
                p1.explore_data(df)
            progress_bar.progress(33)
            
            # Phase 2
            status_text.text("Phase 2/3: Calculating carbon intensity...")
            
            import phase2_carbon_intensity as p2
            df = p2.load_power_plant_data()
            df = p2.map_emission_factors(df)
            df = p2.calculate_plant_emissions(df)
            country_data = p2.calculate_country_carbon_intensity(df)
            
            df.to_csv('data/processed/plants_with_emissions.csv', index=False)
            country_data.to_csv('data/processed/country_carbon_intensity.csv', index=False)
            progress_bar.progress(66)
            
            # Phase 3
            status_text.text("Phase 3/3: Training ML emulator...")
            
            import phase3_ml_emulator as p3
            plants_df, country_df = p2.load_power_plant_data(), country_data
            fuel_features = p3.create_fuel_mix_features(plants_df)
            X, y, ml_data = p3.prepare_ml_dataset(fuel_features, country_df)
            
            model_result = p3.train_models(X, y)
            if model_result[0] is not None:
                best_model = model_result[0]
                joblib.dump(best_model, 'data/processed/carbon_emulator_model.pkl')
                X.to_csv('data/processed/ml_features.csv')
                y.to_csv('data/processed/ml_targets.csv')
            
            progress_bar.progress(100)
            status_text.text("Setup complete!")
            st.success("Data pipeline completed successfully! Reloading app...")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during setup: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# Run setup check
setup_data()

# Page configuration
st.set_page_config(
    page_title="Grid Carbon Emulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Climate Tech Design
st.markdown("""
    <style>
    /* Global Settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background - Very Light Mint */
    .stApp {
        background-color: #F0FDF4;
    }
    
    /* Sidebar - White with border */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D1FAE5;
    }
    
    /* Headers - Deep Emerald Green */
    h1 {
        color: #064E3B; /* Emerald 900 */
        font-weight: 800;
        letter-spacing: -0.5px;
        padding-bottom: 1rem;
        border-bottom: 2px solid #10B981; /* Emerald 500 */
        font-size: 2.5rem !important;
    }
    
    h2 {
        color: #065F46; /* Emerald 800 */
        font-weight: 700;
        margin-top: 2rem;
        font-size: 1.8rem !important;
    }
    
    h3 {
        color: #047857; /* Emerald 700 */
        font-weight: 600;
        font-size: 1.4rem !important;
    }
    
    /* Metrics Cards - Modern & Clean */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
        color: #059669; /* Emerald 600 */
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #4B5563; /* Gray 600 */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #E5E7EB;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.1rem;
        color: #4B5563;
        padding: 1rem 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #059669;
        border-bottom-color: #059669;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #059669;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        border: none;
        font-size: 1rem;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #047857;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
    }
    
    /* Input Fields */
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        border-color: #D1FAE5;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        background-color: white;
    }
    
    /* Alert Boxes */
    .stAlert {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #064E3B;
    }
    
    /* Footer */
    footer {
        color: #6B7280;
        text-align: center;
        padding: 3rem 0;
        border-top: 1px solid #E5E7EB;
        margin-top: 4rem;
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    """Load processed data"""
    country_data = pd.read_csv('data/processed/country_carbon_intensity.csv')
    ml_features = pd.read_csv('data/processed/ml_features.csv', index_col=0)
    ml_targets = pd.read_csv('data/processed/ml_targets.csv', index_col=0)
    
    return country_data, ml_features, ml_targets

@st.cache_resource
def load_model():
    """Load trained model"""
    return joblib.load('data/processed/carbon_emulator_model.pkl')

# Load everything
try:
    country_data, ml_features, ml_targets = load_data()
    model = load_model()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure you've run Phase 2 and Phase 3 first!")
    data_loaded = False

if data_loaded:
    # Title and description
    st.title("Grid Carbon-Intensity Emulator")
    st.markdown("""
    <p style='font-size: 1.25rem; color: #374151; margin-bottom: 2.5rem; line-height: 1.6;'>
    AI-powered analysis and simulation of electricity grid carbon intensity across 167 countries. 
    Predict the impact of energy policy changes and optimize data center location strategies.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("Use the tabs above to explore different features of the emulator.")
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["Global Overview", "Policy Simulator", "Country Analysis", "Data Center Calculator"])
    
    # TAB 1: Global Overview
    with tab1:
        st.markdown("## Global Carbon Intensity Map")
        st.markdown("Explore carbon intensity patterns across countries and identify leaders in clean energy.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_intensity = country_data['carbon_intensity_gco2_kwh'].mean()
            st.metric("Global Average", f"{avg_intensity:.0f} gCO2/kWh")
        
        with col2:
            cleanest = country_data.nsmallest(1, 'carbon_intensity_gco2_kwh').iloc[0]
            st.metric("Cleanest Grid", f"{cleanest['country']}: {cleanest['carbon_intensity_gco2_kwh']:.0f}")
        
        with col3:
            dirtiest = country_data.nlargest(1, 'carbon_intensity_gco2_kwh').iloc[0]
            st.metric("Dirtiest Grid", f"{dirtiest['country']}: {dirtiest['carbon_intensity_gco2_kwh']:.0f}")
        
        with col4:
            avg_renewable = country_data['renewable_percentage'].mean()
            st.metric("Avg Renewable %", f"{avg_renewable:.1f}%")
        
        # World map
        st.subheader("Carbon Intensity by Country")
        
        fig = px.choropleth(
            country_data,
            locations='country',
            locationmode='ISO-3',
            color='carbon_intensity_gco2_kwh',
            hover_name='country',
            hover_data={
                'carbon_intensity_gco2_kwh': ':.0f',
                'renewable_percentage': ':.1f',
                'dominant_fuel': True,
                'country': False
            },
            color_continuous_scale='RdYlGn_r', # Red to Green (Green is good/low carbon)
            labels={'carbon_intensity_gco2_kwh': 'Carbon Intensity (gCO2/kWh)'}
        )
        
        fig.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(0,0,0,0)', showframe=False, projection_type='natural earth')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top/Bottom countries
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cleanest Grids")
            cleanest_10 = country_data.nsmallest(10, 'carbon_intensity_gco2_kwh')[
                ['country', 'carbon_intensity_gco2_kwh', 'renewable_percentage', 'dominant_fuel']
            ]
            cleanest_10.columns = ['Country', 'gCO2/kWh', 'Renewable %', 'Dominant Fuel']
            st.dataframe(cleanest_10, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Highest Emitters")
            dirtiest_10 = country_data.nlargest(10, 'carbon_intensity_gco2_kwh')[
                ['country', 'carbon_intensity_gco2_kwh', 'renewable_percentage', 'dominant_fuel']
            ]
            dirtiest_10.columns = ['Country', 'gCO2/kWh', 'Renewable %', 'Dominant Fuel']
            st.dataframe(dirtiest_10, hide_index=True, use_container_width=True)
    
    # TAB 2: Policy Simulator
    with tab2:
        st.markdown("## Policy Impact Simulator")
        st.markdown("Model the effects of transitioning from fossil fuels to renewable energy sources.")
        
        # Country selection
        countries_with_coal = ml_features[ml_features.get('Coal_pct', 0) > 5].index.tolist()
        
        if len(countries_with_coal) > 0:
            selected_country = st.sidebar.selectbox(
                "Select Country (Policy Sim)",
                countries_with_coal,
                index=0 if 'USA' not in countries_with_coal else countries_with_coal.index('USA')
            )
            
            # Get baseline
            baseline_features = ml_features.loc[[selected_country]].copy()
            baseline_intensity = model.predict(baseline_features)[0]
            
            # Display current state
            st.subheader(f"Current State: {selected_country}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Carbon Intensity", f"{baseline_intensity:.0f} gCO2/kWh")
            
            with col2:
                coal_pct = baseline_features.get('Coal_pct', pd.Series([0])).values[0]
                st.metric("Coal Percentage", f"{coal_pct:.1f}%")
            
            with col3:
                renewable_ratio = baseline_features.get('renewable_ratio', pd.Series([0])).values[0]
                st.metric("Renewable Ratio", f"{renewable_ratio*100:.1f}%")
            
            # Scenario sliders
            st.subheader("Adjust Energy Mix")
            
            coal_reduction = st.slider(
                "Replace Coal with Solar (%)",
                min_value=0,
                max_value=min(100, int(coal_pct)),
                value=0,
                step=5,
                help="Percentage of coal capacity to replace with solar"
            )
            
            # Calculate scenario
            scenario_features = baseline_features.copy()
            
            if coal_reduction > 0 and 'Coal_pct' in scenario_features.columns and 'Solar_pct' in scenario_features.columns:
                scenario_features['Coal_pct'] -= coal_reduction
                scenario_features['Solar_pct'] += coal_reduction
                
                # Update renewable ratio
                renewables = ['Solar', 'Wind', 'Hydro', 'Geothermal']
                renewable_cols = [f'{r}_pct' for r in renewables if f'{r}_pct' in scenario_features.columns]
                new_renewable_pct = scenario_features[renewable_cols].sum(axis=1).values[0]
                scenario_features['renewable_ratio'] = new_renewable_pct / 100
                
                # Update fossil ratio
                fossils = ['Coal', 'Oil', 'Gas']
                fossil_cols = [f'{f}_pct' for f in fossils if f'{f}_pct' in scenario_features.columns]
                new_fossil_pct = scenario_features[fossil_cols].sum(axis=1).values[0]
                scenario_features['fossil_ratio'] = new_fossil_pct / 100
            
            scenario_intensity = model.predict(scenario_features)[0]
            reduction = baseline_intensity - scenario_intensity
            reduction_pct = (reduction / baseline_intensity) * 100
            
            # Results
            st.subheader("Scenario Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "New Carbon Intensity",
                    f"{scenario_intensity:.0f} gCO2/kWh",
                    delta=f"-{reduction:.0f} gCO2/kWh",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Reduction",
                    f"{reduction_pct:.1f}%",
                    help="Percentage reduction in carbon intensity"
                )
            
            with col3:
                # Estimate CO2 saved (assuming 100 TWh annual generation)
                annual_generation_twh = 100
                co2_saved = reduction * annual_generation_twh * 1_000_000_000 / 1_000_000_000
                st.metric(
                    "CO₂ Saved (Mt/year)",
                    f"{co2_saved:.1f}",
                    help="Assumes 100 TWh annual generation"
                )
            
            # Visualization
            comparison_df = pd.DataFrame({
                'Scenario': ['Current', 'With Transition'],
                'Carbon Intensity (gCO2/kWh)': [baseline_intensity, scenario_intensity]
            })
            
            fig = px.bar(
                comparison_df,
                x='Scenario',
                y='Carbon Intensity (gCO2/kWh)',
                color='Scenario',
                color_discrete_map={'Current': '#9CA3AF', 'With Transition': '#10B981'}, # Grey vs Green
                title='Carbon Intensity Comparison'
            )
            
            fig.update_layout(showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No countries with significant coal capacity found in the dataset.")
    
    # TAB 3: Country Deep Dive
    with tab3:
        st.header("Country Deep Dive")
        
        # Country selector
        all_countries = sorted(country_data['country'].tolist())
        selected_country_dive = st.sidebar.selectbox(
            "Select Country for Analysis",
            all_countries,
            index=all_countries.index('USA') if 'USA' in all_countries else 0
        )
        
        country_info = country_data[country_data['country'] == selected_country_dive].iloc[0]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Carbon Intensity", f"{country_info['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh")
        
        with col2:
            st.metric("Total Capacity", f"{country_info['total_capacity_mw']:,.0f} MW")
        
        with col3:
            st.metric("Renewable %", f"{country_info['renewable_percentage']:.1f}%")
        
        with col4:
            st.metric("Dominant Fuel", country_info['dominant_fuel'])
        
        # Fuel mix breakdown
        if selected_country_dive in ml_features.index:
            st.subheader("Energy Mix Breakdown")
            
            fuel_cols = [col for col in ml_features.columns if col.endswith('_pct')]
            fuel_data = ml_features.loc[selected_country_dive, fuel_cols]
            fuel_data = fuel_data[fuel_data > 0].sort_values(ascending=False)
            
            if len(fuel_data) > 0:
                fuel_df = pd.DataFrame({
                    'Fuel Type': [col.replace('_pct', '') for col in fuel_data.index],
                    'Percentage': fuel_data.values
                })
                
                # Use a professional color sequence
                fig = px.pie(
                    fuel_df,
                    values='Percentage',
                    names='Fuel Type',
                    title=f'{selected_country_dive} Energy Mix',
                    hole=0.5,
                    color_discrete_sequence=px.colors.sequential.Greens_r
                )
                
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fuel mix data available for this country")
        
        # Comparison with global average
        st.subheader("Comparison with Global Average")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Carbon Intensity', 'Renewable %'],
            selected_country_dive: [
                country_info['carbon_intensity_gco2_kwh'],
                country_info['renewable_percentage']
            ],
            'Global Average': [
                country_data['carbon_intensity_gco2_kwh'].mean(),
                country_data['renewable_percentage'].mean()
            ]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=selected_country_dive,
            x=comparison_data['Metric'],
            y=comparison_data[selected_country_dive],
            marker_color='#059669' # Emerald
        ))
        
        fig.add_trace(go.Bar(
            name='Global Average',
            x=comparison_data['Metric'],
            y=comparison_data['Global Average'],
            marker_color='#9CA3AF' # Grey
        ))
        
        fig.update_layout(
            barmode='group',
            title='Country vs Global Average',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Data Center Carbon Calculator
    with tab4:
        st.header("Data Center Carbon Cost Calculator")
        st.markdown("""
        Calculate the carbon footprint and potential carbon tax costs for data centers based on location.
        Understand how grid carbon intensity affects operational sustainability.
        """)
        
        # User inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Center Specifications")
            
            dc_power_mw = st.number_input(
                "Power Capacity (MW)",
                min_value=1,
                max_value=500,
                value=50,
                help="Total power capacity of the data center"
            )
            
            utilization = st.slider(
                "Average Utilization (%)",
                min_value=10,
                max_value=100,
                value=70,
                help="Percentage of capacity typically used"
            )
            
            pue = st.slider(
                "Power Usage Effectiveness (PUE)",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="1.0 = perfect efficiency, industry average ~1.5-2.0"
            )
        
        with col2:
            st.subheader("Location")
            
            dc_country = st.selectbox(
                "Data Center Location",
                options=sorted(country_data['country'].tolist()),
                index=sorted(country_data['country'].tolist()).index('USA') if 'USA' in country_data['country'].tolist() else 0
            )
            
            carbon_tax = st.number_input(
                "Carbon Tax ($/tonne CO₂)",
                min_value=0,
                max_value=200,
                value=50,
                help="Current or proposed carbon tax rate"
            )
            
            st.info(f"EU ETS carbon price: ~€80-100/tonne | Singapore: S$25/tonne (rising to S$80)")
        
        # Get country carbon intensity
        country_info = country_data[country_data['country'] == dc_country].iloc[0]
        carbon_intensity = country_info['carbon_intensity_gco2_kwh']
        
        # Calculate metrics
        effective_power_mw = dc_power_mw * (utilization / 100) * pue
        annual_energy_gwh = effective_power_mw * 8760 / 1000
        annual_emissions_tonnes = (annual_energy_gwh * 1_000_000 * carbon_intensity) / 1_000_000_000
        annual_carbon_cost = annual_emissions_tonnes * carbon_tax
        
        # Results
        st.subheader("Annual Carbon Footprint")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Energy Consumption",
                f"{annual_energy_gwh:,.0f} GWh/year"
            )
        
        with col2:
            st.metric(
                "CO₂ Emissions",
                f"{annual_emissions_tonnes:,.0f} tonnes/year"
            )
        
        with col3:
            st.metric(
                "Grid Intensity",
                f"{carbon_intensity:.0f} gCO2/kWh",
                help=f"{dc_country} grid carbon intensity"
            )
        
        with col4:
            st.metric(
                "Carbon Tax Cost",
                f"${annual_carbon_cost:,.0f}/year",
                delta=f"${annual_carbon_cost/12:,.0f}/month",
                help="Annual carbon tax liability"
            )
        
        # Comparison with other locations
        st.subheader("Location Comparison")
        
        comparison_countries = ['USA', 'CHN', 'IND', 'DEU', 'FRA', 'GBR', 'SGP', 'NOR', 'ISL']
        comparison_data = []
        
        for country in comparison_countries:
            if country in country_data['country'].values:
                c_info = country_data[country_data['country'] == country].iloc[0]
                c_emissions = (annual_energy_gwh * 1_000_000 * c_info['carbon_intensity_gco2_kwh']) / 1_000_000_000
                c_cost = c_emissions * carbon_tax
                
                comparison_data.append({
                    'Country': country,
                    'Carbon Intensity': c_info['carbon_intensity_gco2_kwh'],
                    'Annual Emissions (tonnes)': c_emissions,
                    'Annual Carbon Cost ($)': c_cost,
                    'Renewable %': c_info['renewable_percentage']
                })
        
        comp_df = pd.DataFrame(comparison_data).sort_values('Annual Emissions (tonnes)')
        
        # Visualization
        fig = px.bar(
            comp_df,
            x='Country',
            y='Annual Emissions (tonnes)',
            color='Renewable %',
            color_continuous_scale='Greens', # Single color scale for clean look
            title=f'Annual Emissions for {dc_power_mw}MW Data Center by Location',
            hover_data=['Carbon Intensity', 'Annual Carbon Cost ($)']
        )
        
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings potential
        st.subheader("Potential Savings")
        
        cleanest_location = comp_df.iloc[0]
        current_location = comp_df[comp_df['Country'] == dc_country].iloc[0]
        
        emission_savings = current_location['Annual Emissions (tonnes)'] - cleanest_location['Annual Emissions (tonnes)']
        cost_savings = current_location['Annual Carbon Cost ($)'] - cleanest_location['Annual Carbon Cost ($)']
        
        if emission_savings > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"Moving to {cleanest_location['Country']} could save:")
                st.markdown(f"""
                * **{emission_savings:,.0f} tonnes CO₂/year**
                * **${cost_savings:,.0f}/year** in carbon costs
                * **{(emission_savings/current_location['Annual Emissions (tonnes)'])*100:.1f}%** emission reduction
                """)
            
            with col2:
                ten_year_savings = cost_savings * 10
                st.info(f"10-Year Projection:")
                st.markdown(f"""
                * Total carbon cost savings: **${ten_year_savings:,.0f}**
                * Total emissions avoided: **{emission_savings*10:,.0f} tonnes**
                * Equivalent to taking **{(emission_savings*10/4.6):,.0f} cars** off the road
                """)
        else:
            st.success(f"{dc_country} is already among the cleanest locations for data centers!")
        
        # Renewable energy recommendations
        st.subheader("Carbon Reduction Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**On-site Solutions**")
            st.markdown("""
            * Install solar panels (reduce grid dependency)
            * Battery storage for load shifting
            * Power Purchase Agreements (PPAs) for renewables
            * Improve PUE through cooling optimization
            """)
        
        with col2:
            st.markdown("**Location Strategies**")
            st.markdown("""
            * Prioritize regions with clean grids
            * Consider carbon intensity in site selection
            * Multi-region deployment for workload shifting
            * Partner with utilities on renewable projects
            """)
        
        # Data table
        with st.expander("View Detailed Comparison Table"):
            st.dataframe(
                comp_df.style.format({
                    'Carbon Intensity': '{:.0f}',
                    'Annual Emissions (tonnes)': '{:,.0f}',
                    'Annual Carbon Cost ($)': '${:,.0f}',
                    'Renewable %': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    <strong>Grid Carbon-Intensity Emulator</strong> | Built with Python, XGBoost, and Streamlit<br>
    Data source: Global Power Plant Database by World Resources Institute
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please run Phase 2 and Phase 3 to generate the required data files.")