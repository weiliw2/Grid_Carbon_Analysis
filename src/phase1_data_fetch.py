"""
Grid Carbon-Intensity Emulator - Phase 1
Data Fetching and Exploration
"""

import pandas as pd
import requests
from io import StringIO
import os

def fetch_power_plant_data():
    """Download the Global Power Plant Database"""
    
    print("🌍 Fetching Global Power Plant Database...")
    print("This may take a minute - the file is ~15MB\n")
    
    # CSV URL from WRI GitHub (v1.3.0 - the latest version)
    csv_url = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
    
    try:
        # Download the data
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
        
        # Load into pandas
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Success! Loaded {len(df):,} power plants")
        print(f"📊 Columns available: {len(df.columns)}")
        print(f"   Key columns: {', '.join(df.columns[:8])}\n")
        
        # Save to data/raw
        output_path = 'data/raw/power_plants_global.csv'
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to '{output_path}'")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Try downloading manually from:")
        print("      https://github.com/wri/global-power-plant-database")
        print("   3. Save it as 'data/raw/power_plants_global.csv'")
        return None

def explore_data(df):
    """Quick exploration of the dataset"""
    
    if df is None:
        return
    
    print("\n" + "="*70)
    print("GLOBAL DATASET OVERVIEW")
    print("="*70)
    
    # Basic stats
    print(f"\n🌐 Countries represented: {df['country'].nunique()}")
    print(f"⚡ Unique fuel types: {df['primary_fuel'].nunique()}")
    print(f"🏭 Total global capacity: {df['capacity_mw'].sum():,.0f} MW")
    print(f"📍 Plants with location data: {df['latitude'].notna().sum():,} ({df['latitude'].notna().sum()/len(df)*100:.1f}%)")
    
    # Top countries by capacity
    print("\n🏆 Top 10 Countries by Total Capacity:")
    top_countries = df.groupby('country')['capacity_mw'].sum().sort_values(ascending=False).head(10)
    for i, (country, capacity) in enumerate(top_countries.items(), 1):
        print(f"   {i:2d}. {country}: {capacity:,.0f} MW")
    
    # Top fuel types globally
    print("\n🔥 Global Fuel Mix (by capacity):")
    fuel_capacity = df.groupby('primary_fuel')['capacity_mw'].sum().sort_values(ascending=False).head(10)
    for fuel, capacity in fuel_capacity.items():
        percentage = (capacity / df['capacity_mw'].sum()) * 100
        print(f"   {fuel:15s}: {capacity:>10,.0f} MW ({percentage:5.1f}%)")
    
    # USA deep dive
    print("\n" + "="*70)
    print("USA DEEP DIVE (Example)")
    print("="*70)
    
    usa_plants = df[df['country'] == 'USA']
    print(f"\n🇺🇸 USA has {len(usa_plants):,} power plants")
    print(f"   Total USA capacity: {usa_plants['capacity_mw'].sum():,.0f} MW")
    
    fuel_summary = usa_plants.groupby('primary_fuel').agg({
        'capacity_mw': ['sum', 'count']
    }).round(2)
    fuel_summary.columns = ['Total Capacity (MW)', 'Number of Plants']
    fuel_summary = fuel_summary.sort_values('Total Capacity (MW)', ascending=False)
    
    print("\n⚡ USA Fuel Breakdown:")
    for fuel in fuel_summary.head(10).index:
        capacity = fuel_summary.loc[fuel, 'Total Capacity (MW)']
        count = int(fuel_summary.loc[fuel, 'Number of Plants'])
        percentage = (capacity / usa_plants['capacity_mw'].sum()) * 100
        print(f"   {fuel:15s}: {capacity:>10,.0f} MW ({count:>5,} plants, {percentage:5.1f}%)")

def data_quality_check(df):
    """Check data quality and missing values"""
    
    if df is None:
        return
    
    print("\n" + "="*70)
    print("DATA QUALITY CHECK")
    print("="*70)
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    important_cols = ['country', 'primary_fuel', 'capacity_mw', 'latitude', 'longitude', 
                      'estimated_generation_gwh', 'estimated_generation_gwh_2017', 
                      'estimated_generation_gwh_2018', 'estimated_generation_gwh_2019']
    
    for col in important_cols:
        if col in df.columns:
            print(f"   {col:30s}: {missing[col]:>6,} missing ({missing_pct[col]:>5.1f}%)")
    
    print("\n✨ Key Insights:")
    print(f"   • {(df['capacity_mw'] > 0).sum():,} plants have capacity data")
    
    # Check for generation columns (name might vary)
    gen_cols = [col for col in df.columns if 'generation' in col.lower()]
    if gen_cols:
        gen_col = gen_cols[0]
        print(f"   • {df[gen_col].notna().sum():,} plants have generation estimates")
    
    print(f"   • {len(df[df['primary_fuel'].str.contains('Solar|Wind|Hydro', case=False, na=False)]):,} renewable energy plants")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRID CARBON-INTENSITY EMULATOR")
    print("Phase 1: Global Data Engineering")
    print("="*70 + "\n")
    
    # Check if data already exists
    data_path = 'data/raw/power_plants_global.csv'
    
    if os.path.exists(data_path):
        print(f"📂 Found existing data at '{data_path}'")
        print("   Loading from disk (faster)...\n")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} power plants from local file")
    else:
        # Fetch data
        df = fetch_power_plant_data()
    
    if df is not None:
        # Explore it
        explore_data(df)
        
        # Quality check
        data_quality_check(df)
        
        print("\n" + "="*70)
        print("✅ PHASE 1 COMPLETE!")
        print("="*70)
        print("\n📋 Next Steps:")
        print("   1. Commit this work: git add . && git commit -m 'Phase 1: Data fetching complete'")
        print("   2. Phase 2: Calculate carbon intensity for each plant")
        print("   3. Phase 3: Aggregate to country-level features")
        print("\n💡 The data is now ready in 'data/raw/power_plants_global.csv'")
        print("="*70 + "\n")
    else:
        print("\n❌ Could not load data. Please check the error messages above.")