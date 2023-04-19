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
IGNORE_LOCKS = os.getenv('IGNORE_LOCKS',False)
ONLY_PRIMARY = os.getenv('ONLY_PRIMARY',True)

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

models = list(collection.find({}).sort('id',-1))

# Make a discard directory for redundant mode formats (.ckpt)
os.makedirs(f"models/discard", exist_ok=True)
# New location
os.makedirs(f"models/files", exist_ok=True)

for model in models:
    if "modelVersions" in model:
        for version in model["modelVersions"]:
            os.makedirs(f"models/{model['id']}/{version['id']}", exist_ok=True)
            with open(f"models/{model['id']}/{version['id']}/modelVersion.json", "w") as f:
                json.dump(json.loads(json_util.dumps(version)), f, indent=4)
            if "files" in version:
                for file in version["files"]:
                    fileDirectory = f"models/{model['id']}/{version['id']}/{file['id']}"
                    fileName = f"{fileDirectory}/{file['name']}"
                    hashFileName = f"{fileDirectory}/{file['name']}.sha256"
                    
                    if os.path.isfile(fileName) and os.path.isfile(hashFileName):
                        try:
                            h = file['hashes']['SHA256'].lower()
                            new_filename = f"models/files/{h}"
                            print(f"✅ Moving {fileName} to {new_filename}...")
                            os.rename(fileName, new_filename)
                        except:
                            tb = traceback.format_exc()
                            print(f"🛑 Problem moving {fileName} to {new_filename}...")
