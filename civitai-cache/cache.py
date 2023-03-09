import requests, os, json, traceback, hashlib
from bson import json_util
from pymongo import MongoClient
from tqdm import tqdm  # tqdm is a package for progress bar
from dotenv import load_dotenv

load_dotenv()

hash_sha256 = hashlib.sha256()

# set up a MongoDB client and database
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION',"mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE',"database")
MODELS_COLLECTION = os.getenv('MODELS_COLLECTION',"models")
CHECK_HASH = os.getenv('CHECK_HASH',False)

client = MongoClient(MONGODB_CONNECTION)
db = client[MONGODB_DATABASE]
collection = db[MODELS_COLLECTION]

def sha256(filename):
    """
    Returns the SHA256 hash of the given file.
    """
    hash_sha256 = hashlib.sha256()

    with open(filename, "rb") as f:
        # Read the file in chunks of 4KB
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)

    # Get the hexadecimal representation of the hash
    return hash_sha256.hexdigest()

models = list(collection.find({}))
for model in models:
    os.makedirs(f"models/{model['id']}", exist_ok=True)
    try:
        with open(f"models/{model['id']}/model.json", "w") as f:
            json.dump(json.loads(json_util.dumps(model)), f, indent=4)
        if "modelVersions" in model:
            for version in model["modelVersions"]:
                os.makedirs(f"models/{model['id']}/{version['id']}", exist_ok=True)
                with open(f"models/{model['id']}/{version['id']}/modelVersion.json", "w") as f:
                    json.dump(json.loads(json_util.dumps(version)), f, indent=4)
                if "files" in version:
                    for file in version["files"]:
                        fileDirectory = f"models/{model['id']}/{version['id']}/{file['id']}"
                        os.makedirs(fileDirectory, exist_ok=True)
                        with open(f"{fileDirectory}/file.json", "wb") as f:
                            f.write(json_util.dumps(file).encode('utf-8'))
                            fileName = f"{fileDirectory}/{file['name']}"
                            download = False
                            if os.path.isfile(fileName):
                                print(f"File {fileName} already exists.")
                                if CHECK_HASH:
                                    print(f"Checking SHA256 hash...")
                                    h = sha256(fileName).lower()
                                    s = file['hashes']['SHA256'].lower()
                                    if s != h:
                                        print(f"🛑 Hash {s} expected but got {h}.  Will redownload...")
                                        download = True
                                    else:
                                        print("✅ Hash good.")
                            else:
                                download = True
                            
                            if download:
                                print(f"Downloading {fileName}")
                                # Send a request to the URL with streaming enabled
                                response = requests.get(file["downloadUrl"], stream=True)
                                # Get the total file size from the response headers
                                total_size = int(response.headers.get('content-length', 0))
                                # Define a chunk size (in bytes) for each iteration
                                chunk_size = 1024
                                # Create a progress bar
                                progress = tqdm(total=total_size, unit='B', unit_scale=True)
                                # Write the response content to a file in chunks
                                with open(fileName, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=chunk_size):
                                        # Write the chunk to the file
                                        f.write(chunk)
                                        
                                        # Update the progress bar
                                        progress.update(len(chunk))

                                # Close the progress bar
                                progress.close()
                                print(f'{fileName} has been downloaded successfully.')
    except:
        tb = traceback.format_exc()
        print(f"Could not save {model['id']}\n{tb}")
