"""
Grid Carbon-Intensity Emulator - Phase 2
Carbon Intensity Calculations
"""

import pandas as pd
import numpy as np

# Emission Factors (gCO2/kWh) - Standard values from IPCC and EPA
EMISSION_FACTORS = {
    'Coal': 820,
    'Oil': 650,
    'Gas': 490,
    'Petcoke': 900,
    'Cogeneration': 400,  # Assuming gas-based
    'Biomass': 230,  # Carbon neutral debated, using lifecycle emissions
    'Nuclear': 12,  # Lifecycle emissions
    'Hydro': 24,  # Lifecycle emissions from construction
    'Wind': 11,  # Lifecycle emissions
    'Solar': 48,  # Lifecycle emissions (panels manufacturing)
    'Geothermal': 38,  # Lifecycle emissions
    'Wave and Tidal': 15,  # Estimated
    'Storage': 0,  # Storage doesn't generate, just stores
    'Other': 500,  # Conservative estimate
    'Waste': 700,  # Municipal waste incineration
}

def load_power_plant_data():
    """Load the data from Phase 1"""
    print("📂 Loading Global Power Plant Database...")
    df = pd.read_csv('data/raw/power_plants_global.csv')
    print(f"✅ Loaded {len(df):,} power plants\n")
    return df

def map_emission_factors(df):
    """Map emission factors to each fuel type"""
    
    print("🔬 Mapping emission factors to fuel types...")
    
    # Create emission factor column
    df['emission_factor_gco2_kwh'] = df['primary_fuel'].map(EMISSION_FACTORS)
    
    # Handle unmapped fuel types
    unmapped = df[df['emission_factor_gco2_kwh'].isna()]['primary_fuel'].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️  Unmapped fuel types found: {list(unmapped)}")
        print(f"   Setting these to 'Other' category (500 gCO2/kWh)")
        df['emission_factor_gco2_kwh'].fillna(EMISSION_FACTORS['Other'], inplace=True)
    
    print(f"✅ Emission factors mapped for all {len(df):,} plants\n")
    
    # Show emission factor distribution
    print("📊 Emission Factor Summary:")
    ef_summary = df.groupby('primary_fuel')['emission_factor_gco2_kwh'].first().sort_values()
    for fuel, ef in ef_summary.items():
        print(f"   {fuel:20s}: {ef:>4.0f} gCO2/kWh")
    
    return df

def calculate_plant_emissions(df):
    """Calculate annual emissions for each plant"""
    
    print("\n⚡ Calculating plant-level emissions...")
    
    # Find generation columns
    gen_cols = [col for col in df.columns if 'generation' in col.lower() and 'gwh' in col.lower()]
    
    if gen_cols:
        # Use the most recent year available
        gen_col = sorted(gen_cols)[-1]  # Get most recent year
        print(f"   Using generation data from: {gen_col}")
        
        # Calculate emissions (GWh * 1,000,000 kWh/GWh * gCO2/kWh / 1,000,000,000 = tonnes CO2)
        df['annual_emissions_tonnes'] = (
            df[gen_col] * 1_000_000 * df['emission_factor_gco2_kwh'] / 1_000_000_000
        )
    else:
        # Estimate generation from capacity
        # Assume average capacity factor of 0.5 and 8760 hours/year
        print("   No generation data - estimating from capacity")
        print("   Assumption: 50% capacity factor, 8760 hours/year")
        
        df['estimated_generation_gwh'] = df['capacity_mw'] * 0.5 * 8760 / 1000
        df['annual_emissions_tonnes'] = (
            df['estimated_generation_gwh'] * 1_000_000 * 
            df['emission_factor_gco2_kwh'] / 1_000_000_000
        )
    
    # Remove negative or invalid emissions
    df['annual_emissions_tonnes'] = df['annual_emissions_tonnes'].clip(lower=0)
    
    total_emissions = df['annual_emissions_tonnes'].sum()
    print(f"\n🌍 Global total emissions: {total_emissions:,.0f} tonnes CO2/year")
    print(f"   That's {total_emissions/1_000_000_000:.2f} gigatonnes CO2/year")
    
    return df

def calculate_country_carbon_intensity(df):
    """Aggregate to country-level carbon intensity"""
    
    print("\n🌐 Calculating country-level carbon intensity...")
    
    # Find generation column
    gen_cols = [col for col in df.columns if 'generation' in col.lower() and 'gwh' in col.lower()]
    gen_col = sorted(gen_cols)[-1] if gen_cols else 'estimated_generation_gwh'
    
    # Aggregate by country
    country_data = df.groupby('country').agg({
        'capacity_mw': 'sum',
        gen_col: 'sum',
        'annual_emissions_tonnes': 'sum',
        'name': 'count'
    }).reset_index()
    
    country_data.columns = ['country', 'total_capacity_mw', 'total_generation_gwh', 
                            'total_emissions_tonnes', 'num_plants']
    
    # Calculate carbon intensity (gCO2/kWh)
    country_data['carbon_intensity_gco2_kwh'] = (
        country_data['total_emissions_tonnes'] * 1_000_000_000 / 
        (country_data['total_generation_gwh'] * 1_000_000)
    )
    
    # Calculate renewable percentage
    renewables = ['Solar', 'Wind', 'Hydro', 'Geothermal', 'Wave and Tidal']
    renewable_capacity = df[df['primary_fuel'].isin(renewables)].groupby('country')['capacity_mw'].sum()
    country_data['renewable_capacity_mw'] = country_data['country'].map(renewable_capacity).fillna(0)
    country_data['renewable_percentage'] = (
        country_data['renewable_capacity_mw'] / country_data['total_capacity_mw'] * 100
    )
    
    # Get dominant fuel type per country
    dominant_fuel = df.groupby('country').apply(
        lambda x: x.groupby('primary_fuel')['capacity_mw'].sum().idxmax()
    )
    country_data['dominant_fuel'] = country_data['country'].map(dominant_fuel)
    
    print(f"✅ Processed {len(country_data)} countries\n")
    
    # Sort by carbon intensity
    country_data = country_data.sort_values('carbon_intensity_gco2_kwh', ascending=False)
    
    return country_data

def show_insights(country_data):
    """Display key insights"""
    
    print("="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print("\n🏆 TOP 10 CLEANEST GRIDS (Lowest Carbon Intensity):")
    cleanest = country_data.nsmallest(10, 'carbon_intensity_gco2_kwh')
    for i, row in enumerate(cleanest.itertuples(), 1):
        print(f"   {i:2d}. {row.country:20s}: {row.carbon_intensity_gco2_kwh:>6.0f} gCO2/kWh "
              f"({row.renewable_percentage:>5.1f}% renewable, {row.dominant_fuel})")
    
    print("\n🚨 TOP 10 DIRTIEST GRIDS (Highest Carbon Intensity):")
    dirtiest = country_data.nlargest(10, 'carbon_intensity_gco2_kwh')
    for i, row in enumerate(dirtiest.itertuples(), 1):
        print(f"   {i:2d}. {row.country:20s}: {row.carbon_intensity_gco2_kwh:>6.0f} gCO2/kWh "
              f"({row.renewable_percentage:>5.1f}% renewable, {row.dominant_fuel})")
    
    print("\n🌍 MAJOR ECONOMIES:")
    major = ['USA', 'CHN', 'IND', 'DEU', 'GBR', 'FRA', 'JPN', 'BRA']
    major_data = country_data[country_data['country'].isin(major)]
    for row in major_data.itertuples():
        print(f"   {row.country:5s}: {row.carbon_intensity_gco2_kwh:>6.0f} gCO2/kWh "
              f"({row.renewable_percentage:>5.1f}% renewable, {row.dominant_fuel})")
    
    print("\n💡 Global Averages:")
    avg_intensity = (country_data['total_emissions_tonnes'].sum() * 1_000_000_000 / 
                     (country_data['total_generation_gwh'].sum() * 1_000_000))
    avg_renewable = (country_data['renewable_capacity_mw'].sum() / 
                     country_data['total_capacity_mw'].sum() * 100)
    print(f"   Average carbon intensity: {avg_intensity:.0f} gCO2/kWh")
    print(f"   Global renewable percentage: {avg_renewable:.1f}%")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRID CARBON-INTENSITY EMULATOR")
    print("Phase 2: Carbon Intensity Calculations")
    print("="*70 + "\n")
    
    # Load data
    df = load_power_plant_data()
    
    # Map emission factors
    df = map_emission_factors(df)
    
    # Calculate plant emissions
    df = calculate_plant_emissions(df)
    
    # Calculate country-level metrics
    country_data = calculate_country_carbon_intensity(df)
    
    # Show insights
    show_insights(country_data)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    df.to_csv('data/processed/plants_with_emissions.csv', index=False)
    country_data.to_csv('data/processed/country_carbon_intensity.csv', index=False)
    print("   ✅ Saved to 'data/processed/plants_with_emissions.csv'")
    print("   ✅ Saved to 'data/processed/country_carbon_intensity.csv'")
    
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("   1. Commit: git add . && git commit -m 'Phase 2: Carbon intensity calculations'")
    print("   2. Phase 3: Build the ML emulator")
    print("   3. Review the processed data in data/processed/")
    print("="*70 + "\n")