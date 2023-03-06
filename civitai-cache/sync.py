import requests, os
from pymongo import MongoClient

# set up a MongoDB client and database
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION',"mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE',"database")
MODELS_COLLECTION = os.getenv('MODELS_COLLECTION',"models")

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
    response = requests.get(url)
    results = response.json()
    # print(results)
    totalPages = results['metadata']['totalPages']
    print(f"{totalPages - page} pages to go...")
    # break the loop if there are no more results
    if not response.json():
        break

    # insert the results into the MongoDB collection
    collection.insert_many(results["items"])

    # increment the page number for the next iteration
    page += 1
