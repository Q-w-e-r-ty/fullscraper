import json
import glob

# Function to merge JSON files
def merge_json_files(input_folder, output_file):
    unique_articles = {}
    
    # Get all JSON files in the input folder
    json_files = glob.glob(f"{input_folder}/*.json")
    
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            for article in data:
                url = article.get("url")
                if url and url not in unique_articles:
                    unique_articles[url] = article
    
    # Write the merged data to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(unique_articles.values()), f, indent=4, ensure_ascii=True)

# Example usage
merge_json_files("E:/Padhai/Stock Prediction/news-fetch-master/fullscraper/Infosys", "E:/Padhai/Stock Prediction/news-fetch-master/fullscraper/combined/infosys_combined.json")