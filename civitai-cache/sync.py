import requests, os, json, datetime
from pymongo import MongoClient, UpdateOne

# set up a MongoDB client and database
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION',"mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE',"database")
MODELS_COLLECTION = os.getenv('MODELS_COLLECTION',"models")

client = MongoClient(MONGODB_CONNECTION)
db = client[MONGODB_DATABASE]
collection = db[MODELS_COLLECTION]

# Back up previous model collection
model_list = list(collection.find({}))
backupDir = 'models/backups'
os.makedirs(backupDir, exist_ok=True)
# Get the current date and time
now = datetime.datetime.now()
# Format the timestamp as a string
timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')
# Use the timestamp string as a directory name
backup_name = f'backup_{timestamp_str}'
print(f"Backing up '{MODELS_COLLECTION}' to '{backup_name}'...")
with open(f'{backupDir}/{backup_name}.json', 'w') as f:
    json.dump(model_list, f, default=str)

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
