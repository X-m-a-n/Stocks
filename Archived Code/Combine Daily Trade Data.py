import polars as pl
import numpy as np
import pathlib

def load_stock_data(data_folder: str) -> pl.DataFrame:
    """
    Load all stock data files from a folder into a single dataframe
    
    Parameters:
    data_folder (str): Path to folder containing stock data files
    
    Returns:
    polars.DataFrame: Combined stock data
    """
    # Get all CSV files in folder
    data_path = pathlib.Path(data_folder)
    data_files = list(data_path.glob('*.csv'))
    
    if not data_files:
        raise ValueError(f"No CSV files found in {data_folder}")
        
    # Read and combine all files
    dfs = []
    for file in data_files:
        try:
            df = pl.read_csv(
                file,
                try_parse_dates=True,
                columns=[
                    'date', 'symbol', 'last_price', 'closing_price', 
                    'price_change', 'bid', 'ask', 'volume', 'daily_range_low', 
                    'daily_range_high', 'year_range_low', 'year_range_high'
                ]
            )
            
            # Convert columns to appropriate types
            df = df.with_columns([
                pl.col(['last_price', 'closing_price', 'price_change', 'bid', 'ask', 
                       'daily_range_low', 'daily_range_high', 'year_range_low', 
                       'year_range_high']).cast(pl.Float64),
                pl.col('volume').cast(pl.Int64)
            ])
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files could be processed")
        
    # Combine all dataframes
    combined_df = pl.concat(dfs)
    
    total_rows = len(combined_df)
    
    print(f"Loaded {len(data_files)} files with {total_rows} total rows")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

    # Removing duplicates
    print(f'Removing duplicates...')
    combined_df = combined_df.unique()
    print(f'\n{total_rows - len(combined_df)} duplicates removed {len(combined_df)} unique values remain')
    
    # Sort by date and symbol
    combined_df = combined_df.sort(['date', 'symbol'])

    return combined_df



def main():
    try:
        # Load data
        data_folder = r'C:\Users\michaelsjo\Desktop\Stocks\Data\eod_trade_summary'
        df = load_stock_data(data_folder)
        df.write_csv(r'C:\Users\michaelsjo\Desktop\Stocks\Data\combined_trade_data.csv')

        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()