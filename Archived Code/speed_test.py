import newspaper
import time
from pathlib import Path
import multiprocessing
from functools import partial

def get_files(folderLoc):
    '''This function accepts a list of folders and returns a list of html file paths'''
    filesOnly = []
    
    for folder in folderLoc:
        folder = Path(folder)
        folderFiles = list(folder.glob('**/*.html'))
        # Filter out folders and keep only files
        for f in folderFiles:
            if f.is_file():
                filesOnly.append(f)
    
    return filesOnly

# Original For Loop Implementation
def get_sentiment_original(files):
    # Initialize newspaper
    testUrl = 'https://www.jamaicaobserver.com/2025/04/27/fa-cup-glory-wont-salvage-man-citys-troubled-season-guardiola'
    article = newspaper.Article(testUrl)
    
    articles = []
    
    # Process files with a for loop
    for file in files[:100]:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
            
        # Set the HTML content directly
        article.html = html_content
        article.download_state = 2
        article.parse()
        article.nlp()
        
        articles.append(article.text)
    
    return articles

# Map Function Implementation
def process_file_with_article(file, shared_article):
    """Process a single file using a shared article object"""
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    # Set the HTML content directly
    shared_article.html = html_content
    shared_article.download_state = 2
    shared_article.parse()
    shared_article.nlp()
    
    return shared_article.text

def get_sentiment_map(files):
    # Initialize newspaper
    testUrl = 'https://www.jamaicaobserver.com/2025/04/27/fa-cup-glory-wont-salvage-man-citys-troubled-season-guardiola'
    article = newspaper.Article(testUrl)
    
    # Process files with map
    # Create a partial function with the shared article object
    process_with_shared_article = partial(process_file_with_article, shared_article=article)
    
    # Use map to process files
    articles = list(map(process_with_shared_article, files[:100]))
    
    return articles

# Main test function
def run_tests():
    # Define folder locations
    folders = [
        r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\ICInsider', 
        r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaGleaner',
        r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JIS', 
        r'C:\Users\Joshh\Projects\Stocks\Data\Webscraping\News Scaper\JamaicaObserver\business'
    ]
    
    # Get files
    print("Getting files...")
    all_files = get_files(folders)
    print(f"Found {len(all_files)} files. Testing with first 100.")
    
    if len(all_files) < 100:
        print(f"Warning: Only {len(all_files)} files found, which is less than the 100 requested.")
    
    # Define test functions and their names
    tests = [
        ("Original For Loop Implementation", get_sentiment_original),
        ("Map Function Implementation", get_sentiment_map)
    ]
    
    # Run each test three times for more accurate results
    iterations = 3
    results = {}
    
    for name, func in tests:
        print(f"\nRunning {name}...")
        times = []
        
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")
            start_time = time.time()
            articles = func(all_files)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"  Processed {len(articles)} articles in {elapsed:.2f} seconds")
        
        # Calculate average and stats
        avg_time = sum(times) / len(times)
        results[name] = {
            "times": times,
            "average": avg_time,
            "article_count": len(articles)
        }
    
    # Print summary
    print("\n--- RESULTS SUMMARY ---")
    for name, data in results.items():
        print(f"{name}:")
        print(f"  Average time: {data['average']:.2f} seconds")
        print(f"  All times: {', '.join([f'{t:.2f}s' for t in data['times']])}")
        print(f"  Average time per article: {data['average']/data['article_count']:.4f} seconds")
        print(f"  Throughput: {data['article_count']/data['average']:.2f} articles/second")
    
    # Find the fastest method
    fastest = min(results.items(), key=lambda x: x[1]['average'])
    print(f"\nFastest method: {fastest[0]} with average time of {fastest[1]['average']:.2f} seconds")
    
    # Calculate difference
    for_loop_time = results["Original For Loop Implementation"]["average"]
    map_time = results["Map Function Implementation"]["average"]
    diff = map_time - for_loop_time
    percent = (diff / for_loop_time) * 100
    
    if diff > 0:
        print(f"\nThe Map implementation was {diff:.2f} seconds ({percent:.1f}%) slower than the For Loop")
    else:
        print(f"\nThe Map implementation was {abs(diff):.2f} seconds ({abs(percent):.1f}%) faster than the For Loop")

if __name__ == "__main__":
    run_tests()