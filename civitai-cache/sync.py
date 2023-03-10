import requests, os
from pymongo import MongoClient, UpdateOne

# set up a MongoDB client and database
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION',"mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE',"database")
MODELS_COLLECTION = os.getenv('MODELS_COLLECTION',"models")
CIVITAI_TOKEN = os.getenv('CIVITAI_TOKEN','')

print(MONGODB_CONNECTION)

client = MongoClient(MONGODB_CONNECTION)
db = client[MONGODB_DATABASE]
collection = db[MODELS_COLLECTION]

# loop over pages of the API endpoint and store the results in MongoDB
page = 1
totalPages = 1

while page <= totalPages:
    # make a request to the API endpoint
    url = f'https://civitai.com/api/v1/models?limit=100&page={page}'
    print(f"Fetching {url}...")
    response = requests.get(url=url, headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/517.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    })
    if response.status_code == 200:

        if 'application/json' in response.headers.get('Content-Type', ''):
            results = response.json()
            # print(results)
            totalPages = results['metadata']['totalPages']
            print(f"{totalPages - page} pages to go...")
            # break the loop if there are no more results
            if not response.json():
                break

            # insert the results into the MongoDB collection
            ops = []
            for item in results["items"]:
                ops.append(UpdateOne(
                    {"id": item["id"]},
                    {"$set": item},
                    upsert=True
                ))

            result = collection.bulk_write(ops)

            # increment the page number for the next iteration
            page += 1
        else:
            # Handle non-JSON response
            print(f'Response is not JSON\n{response.text}')

        
    else:
        print('Request failed with status code:', response.status_code)
