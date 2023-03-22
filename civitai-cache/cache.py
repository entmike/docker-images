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

models = list(collection.find({}))

# Make a discard directory for redundant mode formats (.ckpt)
os.makedirs(f"models/discard", exist_ok=True)

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
                        fileName = f"{fileDirectory}/{file['name']}"
                        isPrimary = False
                        skip = False

                        if "primary" in file:
                            if file["primary"] == True:
                                isPrimary = True
                        
                        if ONLY_PRIMARY and not isPrimary:
                            if "format" in file:
                                if file["format"] == "PickleTensor" or file["format"] == "SafeTensor":
                                    skip = True
                                    print(f"{fileName} not the primary model.  Skipping.")
                                

                        if skip:
                            if os.path.isfile(fileName):
                                discardDirectory = f"models/discard/{model['id']}/{version['id']}/{file['id']}"
                                discardFileName = f"{discardDirectory}/{file['name']}"
                                print(f"{fileName} is a redundant format.  Moving to {discardDirectory}")
                                os.makedirs(discardDirectory, exist_ok=True)
                                os.rename(fileName, discardFileName)

                        else:
                            os.makedirs(fileDirectory, exist_ok=True)
                            with open(f"{fileDirectory}/file.json", "wb") as f:
                                f.write(json_util.dumps(file).encode('utf-8'))
                                hashFileName = f"{fileDirectory}/{file['name']}.sha256"
                                download = False
                                if os.path.isfile(fileName):
                                    print(f"File {fileName} already exists.")
                                    if CHECK_HASH:
                                        if os.path.isfile(hashFileName):
                                            print(f"Hash for {fileName} already checked.")
                                        else:
                                            print(f"Checking SHA256 hash...")
                                            h = sha256(fileName).lower()
                                            s = file['hashes']['SHA256'].lower()
                                            if s != h:
                                                print(f"🛑 Hash {s} expected but got {h}.  Will redownload...")
                                                download = True
                                            else:
                                                print(f"✅ Hash good.  Saving to {hashFileName}")
                                                with open(hashFileName, "w") as file:
                                                    # Write the hash to the file
                                                    file.write(h)
                                else:
                                    download = True
                                
                                if download:
                                    lockFileName = f"{fileDirectory}/{file['name']}.lock"
                                    if not os.path.exists(lockFileName):
                                        print(f"Downloading {fileName}")

                                        # Send a request to the URL with streaming enabled
                                        with open(lockFileName, "w") as lockfile:
                                            # Write the hash to the file
                                            lockfile.write("")

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
                                        if os.path.exists(lockFileName):
                                            os.remove(lockFileName)
                                        h = sha256(fileName).lower()
                                        s = file['hashes']['SHA256'].lower()
                                        if s != h:
                                            print(f"🛑 Hash {s} expected but got {h}.  Still...")
                                        else:
                                            print(f"✅ Hash good.  Saving to {hashFileName}")
                                            with open(hashFileName, "w") as file:
                                                # Write the hash to the file
                                                file.write(h)
                                    else:
                                        print(f"{lockFileName} detected.  Skipping download.")
    except:
        tb = traceback.format_exc()
        print(f"Could not save {model['id']}\n{tb}")
