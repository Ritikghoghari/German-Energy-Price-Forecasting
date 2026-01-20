
import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_german_number(x):
    """
    Parses a string with German number formatting (1.234,56) to float.
    Handles already numeric types gracefully.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # Remove dots (thousands separator) and replace comma with dot (decimal separator)
        clean_str = x.replace('.', '').replace(',', '.')
        try:
            return float(clean_str)
        except ValueError:
            return np.nan
    return np.nan

def clean_smard_df(file_path):
    """
    Loads and cleans a SMARD.de CSV file.
    Expects headers in row 0 or handled automatically.
    """
    print(f"Processing {file_path}...")
    
    # Read first few lines to deduce delimiter and structure
    # SMARD files can optionally have metadata at the top, but ours seemed to start with header
    # They often use ';' as delimiter
    
    try:
        df = pd.read_csv(file_path, sep=';', thousands='.', decimal=',')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Identify date columns
    # Common column names seen: 'Date', 'Start date', 'End date', 'timestamp'
    # generation/consumption used 'Start date', 'End date'
    # prices.csv often has simpler structure or matches these
    
    if 'Start date' in df.columns:
        # Combine Date and Time maybe? Or 'Start date' is essentially the unique key with 'Start time'?
        # Actually in the snippet: "Jan 1 2021 12:00 AM" looks like a full datetime string in 'Start date'??
        # Let's inspect the snippet again:
        # Start date;End date;...
        # Jan 1 2021 12:00 AM;Jan 1 2021 1:00 AM;...
        # So 'Start date' holds the full datetime string.
        
        # We need to parse "Jan 1 2021 12:00 AM". 
        # Format seems to be "%b %d %Y %I:%M %p"
        
        df['timestamp'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p', errors='coerce')
        
    elif 'timestamp' in df.columns:
        # Already has timestamp, ensure datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    else:
        # Fallback for prices if different
        # Let's check columns for prices.csv based on earlier error
        pass

    # Drop rows with invalid timestamp
    df = df.dropna(subset=['timestamp'])
    
    # Set index
    df.set_index('timestamp', inplace=True)
    
    # We only care about specific columns, but for now let's keep numeric ones
    # Drop "Start date", "End date"
    cols_to_drop = [c for c in df.columns if 'date' in c.lower() or 'timestamp' in c.lower()]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Rename columns and convert to numeric
    for col in df.columns:
        # Rename columns to be cleaner: remove "[MWh]...", "Calculated..."
        # Example: "Total [MWh] Calculated resolutions" -> "Total_Generation"
        
        clean_name = col.split('[')[0].strip().replace(' ', '_')
        if "residual" in col.lower():
             clean_name = "Residual_Load"
        elif "grid_load" in col.lower() or "grid load" in col.lower():
             clean_name = "Grid_Load"
        
        df.rename(columns={col: clean_name}, inplace=True)
        
        # Convert to numeric if not already
        # The read_csv thousands/decimal might fail if there are spaces or other quirks
        # Apply strict parsing
    
    # Apply numeric conversion to the (now renamed) columns
    for col in df.columns:
        if df[col].dtype == 'object':
             df[col] = df[col].apply(parse_german_number)
    
    return df

def run_pipeline():
    data_dir = '../data'
    src_files = {
        'generation': f'{data_dir}/generation.csv',
        'consumption': f'{data_dir}/consumption.csv',
        'prices': f'{data_dir}/prices.csv'
    }

    dfs = []
    
    # Process Generation
    gen_df = clean_smard_df(src_files['generation'])
    if gen_df is not None:
        # Keep key columns: Total, Wind_offshore, Wind_onshore, Photovoltaics, Other
        # Check actual columns from snippet: 
        # ['Total', 'Photovoltaics_and_wind', 'Wind_offshore', 'Wind_onshore', 'Photovoltaics', 'Other']
        # We probably want specific sources + Total
        dfs.append(gen_df)
        
    # Process Consumption
    cons_df = clean_smard_df(src_files['consumption'])
    if cons_df is not None:
        # Columns: Grid_Load, Residual_Load
        dfs.append(cons_df)

    # Process Prices
    # Prices might have different format. Earlier read failed on sep=';' check?
    # Let's try reading prices with more flexible engine or inspecting first
    # prices = pd.read_csv('prices.csv', sep=';')
    # Let's assume the clean_smard_df handles it or modify it if prices uses different logic
    price_df = clean_smard_df(src_files['prices'])
    if price_df is not None:
        # Rename columns if needed. "Deutschland/Luxemburg [€/MWh] Calculated resolutions"?
        # Let's look for any column with "Deutschland" or "Euro"
        
        found_price_col = False
        for col in price_df.columns:
            if "Deutschland" in col or "€" in col:
                price_df.rename(columns={col: "Day_Ahead_Price"}, inplace=True)
                found_price_col = True
                break
        
        if not found_price_col:
            # Maybe it's just one column?
            if len(price_df.columns) == 1:
                price_df.columns = ["Day_Ahead_Price"]
        
        dfs.append(price_df)

    if not dfs:
        print("No data processed.")
        return

    # Merge all
    print("Merging datasets...")
    # Join on index (timestamp)
    master_df = dfs[0]
    for i in range(1, len(dfs)):
        master_df = master_df.join(dfs[i], how='inner') # Inner join to align timestamps? Or Outer?
        # Inner is safer for modeling to avoid NaNs at edges, assuming valid overlap
    
    # final cleanup
    master_df = master_df.sort_index()
    
    output_path = f'{data_dir}/energy_dataset_master.csv'
    master_df.to_csv(output_path, encoding='utf-8')
    print(f"Saved master dataset to {output_path}")
    
    # Print info instead of head to avoid charmap errors in Windows console with special chars
    print("Master DataFrame Info:")
    master_df.info()
    
    print("\nColumn names:")
    print(master_df.columns.tolist())

if __name__ == "__main__":
    # Set console encoding to utf-8 just in case
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
    run_pipeline()

