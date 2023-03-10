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
        if "modelVersions" in model:
            for version in model["modelVersions"]:
                if "files" in version:
                    for file in version["files"]:
                        fileDirectory = f"models/{model['id']}/{version['id']}/{file['id']}"
                        fileName = f"{fileDirectory}/{file['name']}"
                        hashName = f"{fileDirectory}/{file['name']}.sha256"
                        badHashName = f"{fileDirectory}/{file['name']}.sha256.bad"
                        if os.path.isfile(fileName) and not os.path.isfile(hashName) and not os.path.isfile(badHashName):
                            try:
                                s = file['hashes']['SHA256'].lower()
                            except:
                                s = None

                            if s:    
                                print(f"File {fileName} exists.")
                                print(f"Checking SHA256 hash...")
                                h = sha256(fileName).lower()
                                if s != h:
                                    print(f"🛑 Hash {s} expected but got {h}.  Will need to re-download.  Marking as bad hash.")
                                    with open(badHashName, "w") as file:
                                        # Write the hash to the file
                                        file.write(h)
                                else:
                                    print(f"✅ Hash good.  Saving to {hashName}")
                                    with open(hashName, "w") as file:
                                        # Write the hash to the file
                                        file.write(h)
                            else:
                                print(f"No hash in DB exists for {fileName}.  Skipping.")
    except:
        tb = traceback.format_exc()
        print(f"Could not hash {model['id']}\n{tb}")
