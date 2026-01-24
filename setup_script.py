"""
Setup script to generate data files for Streamlit Cloud deployment
This runs automatically when the app is deployed
"""

import os
import sys

def check_data_files():
    """Check if processed data files exist"""
    required_files = [
        'data/processed/country_carbon_intensity.csv',
        'data/processed/plants_with_emissions.csv',
        'data/processed/carbon_emulator_model.pkl',
        'data/processed/ml_features.csv',
        'data/processed/ml_targets.csv'
    ]
    
    return all(os.path.exists(f) for f in required_files)

def run_data_pipeline():
    """Run the full data pipeline"""
    print("🔄 Data files not found. Running data pipeline...")
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Phase 1: Fetch data
        print("\n📥 Phase 1: Fetching global power plant data...")
        exec(open('src/phase1_data_fetch.py').read())
        
        # Phase 2: Calculate carbon intensity
        print("\n⚡ Phase 2: Calculating carbon intensity...")
        exec(open('src/phase2_carbon_intensity.py').read())
        
        # Phase 3: Train ML model
        print("\n🤖 Phase 3: Training ML emulator...")
        exec(open('src/phase3_ml_emulator.py').read())
        
        print("\n✅ Data pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error running data pipeline: {e}")
        return False

if __name__ == "__main__":
    if not check_data_files():
        success = run_data_pipeline()
        if not success:
            sys.exit(1)
    else:
        print("✅ All data files already exist. Skipping pipeline.")
