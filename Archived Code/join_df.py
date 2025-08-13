import polars as pl
import gc
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("Starting large DataFrame join...")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # File paths
    date_fileloc = r'C:\Users\Joshh\Projects\Stocks\newspaper_dates.parquet'
    fileloc = 'C:/Users/Joshh/Projects/Stocks/Data/newspaper_output.parquet'
    
    try:
        # Load data using lazy evaluation to minimize memory usage
        print("Loading data with lazy evaluation...")
        df_dates_lazy = pl.scan_parquet(date_fileloc)
        df_lazy = pl.scan_parquet(fileloc)
        
        print(f"Memory after loading lazily: {get_memory_usage():.1f} MB")
        
        # Perform the join using lazy evaluation
        print("Performing join...")
        result_lazy = df_lazy.join(
            df_dates_lazy, 
            how='left', 
            left_on='filename', 
            right_on='file_path'
        )
        
        # Collect with new streaming engine to handle large datasets
        print("Collecting results with streaming...")
        try:
            result = result_lazy.collect(engine="streaming")
        except Exception as streaming_error:
            print(f"Streaming engine failed: {streaming_error}")
            print("Falling back to in-memory collection...")
            result = result_lazy.collect()
        
        print(f"Join completed successfully!")
        print(f"Final memory usage: {get_memory_usage():.1f} MB")
        print(f"Result shape: {result.shape}")
        
        # Optional: Save result to avoid recomputation
        output_path = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data.parquet'
        print(f"Saving result to: {output_path}")
        result.write_parquet(output_path)
        
        return result
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Trying alternative approach with chunked processing...")
        
        # Fallback: chunked processing
        try:
            # Load smaller dataset into memory first
            print("Loading dates dataset...")
            df_dates = pl.read_parquet(date_fileloc)
            print(f"Dates dataset shape: {df_dates.shape}")
            
            # Process main dataset in chunks
            chunk_size = 50000
            results = []
            
            # Get total rows first
            total_rows = pl.scan_parquet(fileloc).select(pl.len()).collect().item()
            print(f"Processing {total_rows} rows in chunks of {chunk_size}")
            
            for i in range(0, total_rows, chunk_size):
                print(f"Processing chunk {i//chunk_size + 1}/{(total_rows-1)//chunk_size + 1}")
                
                chunk = pl.scan_parquet(fileloc).slice(i, chunk_size).collect()
                joined_chunk = chunk.join(df_dates, how='left', left_on='filename', right_on='file_path')
                results.append(joined_chunk)
                
                # Clean up memory
                del chunk
                gc.collect()
                
                print(f"Memory usage: {get_memory_usage():.1f} MB")
            
            # Concatenate all results
            print("Concatenating results...")
            final_result = pl.concat(results)
            
            # Clean up
            del results
            gc.collect()
            
            print(f"Chunked join completed!")
            print(f"Final result shape: {final_result.shape}")
            
            # Save result
            output_path = 'C:/Users/Joshh/Projects/Stocks/Data/joined_newspaper_data.parquet'
            print(f"Saving result to: {output_path}")
            final_result.write_parquet(output_path)
            
            return final_result
            
        except Exception as e2:
            print(f"Chunked processing also failed: {str(e2)}")
            print("Consider:")
            print("1. Reducing data size by selecting only needed columns")
            print("2. Adding more RAM to your system")
            print("3. Using a machine with more memory")
            raise

if __name__ == "__main__":
    result = main()