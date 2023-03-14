import requests, os, json, traceback, hashlib
from bson import json_util
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
import blurhash
from PIL import Image
from typing import Dict, Tuple, Union
from io import BytesIO
import boto3, botocore
from datetime import datetime, timedelta

load_dotenv()

hash_sha256 = hashlib.sha256()

# set up a MongoDB client and database
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION',"mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE',"database")
MODELS_COLLECTION = os.getenv('MODELS_COLLECTION',"models")
CHECK_HASH = os.getenv('CHECK_HASH',False)
IGNORE_LOCKS = os.getenv('IGNORE_LOCKS',False)
S3_AWS_SERVER_PUBLIC_KEY = os.getenv('S3_AWS_SERVER_PUBLIC_KEY',None)
S3_AWS_SERVER_SECRET_KEY = os.getenv('S3_AWS_SERVER_SECRET_KEY',None)
S3_BUCKET = os.getenv('S3_BUCKET',None)
S3_OVERWRITE = os.getenv('S3_OVERWRITE',False)

client = MongoClient(MONGODB_CONNECTION)
db = client[MONGODB_DATABASE]
collection = db[MODELS_COLLECTION]

if S3_BUCKET:
    # Create a session with the credentials
    session = boto3.Session(
        aws_access_key_id=S3_AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key=S3_AWS_SERVER_SECRET_KEY
    )
    # Use the session to create an S3 resource
    s3 = session.resource('s3')

def is_file_older_than(file_path, hours):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist")

    modified_time = os.path.getmtime(file_path)
    file_age = datetime.now() - datetime.fromtimestamp(modified_time)

    return file_age > timedelta(hours=hours)

def exists_s3(bucket_name, file_name):
    # Check if the object exists
    try:
        s3.Object(bucket_name, file_name).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True

def upload_file_s3(file_name, bucket, object_name=None, extra_args=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3",
        region_name = "us-east-1",
        aws_access_key_id = S3_AWS_SERVER_PUBLIC_KEY,
        aws_secret_access_key = S3_AWS_SERVER_SECRET_KEY)

    try:
        response = s3_client.upload_file(file_name, bucket, object_name, extra_args)
    except Exception as e:
        logger.error(e)
        return False
    return True

def get_clamped_size(width: int, height: int, max_size: int, clamp_type: str = 'all') -> Tuple[int, int]:
    if clamp_type == 'all':
        if width >= height:
            clamp_type = 'width'
        elif height >= width:
            clamp_type = 'height'

    if clamp_type == 'width' and width > max_size:
        return (max_size, int(height / width * max_size))

    if clamp_type == 'height' and height > max_size:
        return (int(width / height * max_size), max_size)

    return (width, height)

def blur_hash_image(img: Image) -> Dict[str, Union[str, int]]:
    width, height = img.size
    clamped_width, clamped_height = get_clamped_size(width, height, 64)
    # print(clamped_size)
    canvas = Image.new('RGB', (clamped_width, clamped_height))
    resized_img = img.resize((clamped_width, clamped_height))
    canvas.paste(resized_img, (0, 0, clamped_width, clamped_height))
    data = canvas.tobytes()
    # Convert the image to a bytestream
    bytestream = BytesIO()
    canvas.save(bytestream, format='JPEG')
    bytestream.seek(0)
    hash = blurhash.encode(bytestream, x_components=4, y_components=4)
    # Close the bytestream
    bytestream.close()
    return {'hash': hash, 'width': width, 'height': height}


def blurhashFile(filename):
    """
    Returns the blur hash of the given file.
    """
    image = Image.open(filename)
    h = blur_hash_image(image)

    return h["hash"]

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
                if "images" in version:
                    imageDirectory = f"models/{model['id']}/{version['id']}/images"
                    os.makedirs(imageDirectory, exist_ok=True)
                    with open(f"{imageDirectory}/images.json", "wb") as f:
                        f.write(json_util.dumps(version["images"]).encode('utf-8'))
                    
                    for image in version["images"]:
                        url = image['url']
                        imageName = url.split("/")[-1]
                        imageFile = f"{imageDirectory}/{imageName}.jpg"
                        hashFileName = f"{imageFile}.hash"
                        download = False
                        if os.path.isfile(imageFile):
                            print(f"File {imageFile} already exists.")
                            #
                            # I tried, but cannot generate the same blurhash as Civit here: https://github.com/civitai/civitai/blob/8ebc27eadad9feebd7c2e59ceed62544ec34817c/src/utils/blurhash.ts#L75
                            #
                            # if CHECK_HASH:
                            #     if os.path.isfile(hashFileName):
                            #         print(f"Hash for {imageFile} already checked.")
                            #     else:
                            #         print(f"Checking blur hash...")
                            #         h = blurhashFile(imageFile)
                            #         s = image['hash']
                            #         if s != h:
                            #             print(f"🛑 Hash {s} expected but got {h}.  Will redownload...")
                            #             download = True
                            #         else:
                            #             print(f"✅ Hash good.  Saving to {hashFileName}")
                            #             with open(hashFileName, "w") as file:
                            #                 # Write the hash to the file
                            #                 file.write(h)
                        else:
                            download = True
                            
                            if download:
                                lockFileName = f"{imageFile}.lock"
                                if True or not os.path.exists(lockFileName):
                                    print(f"Downloading {imageFile}")

                                    # Send a request to the URL with streaming enabled
                                    with open(lockFileName, "w") as lockfile:
                                        # Write the hash to the file
                                        lockfile.write("")

                                    response = requests.get(image["url"], stream=True)
                                    # Get the total file size from the response headers
                                    total_size = int(response.headers.get('content-length', 0))
                                    # Define a chunk size (in bytes) for each iteration
                                    chunk_size = 1024
                                    # Create a progress bar
                                    progress = tqdm(total=total_size, unit='B', unit_scale=True)
                                    # Write the response content to a file in chunks
                                    with open(imageFile, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=chunk_size):
                                            # Write the chunk to the file
                                            f.write(chunk)
                                            
                                            # Update the progress bar
                                            progress.update(len(chunk))

                                    # Close the progress bar
                                    progress.close()
                                    print(f'{imageFile} has been downloaded successfully.')
                                    if os.path.exists(lockFileName):
                                        os.remove(lockFileName)
                                    
                                    #
                                    # I tried, but cannot generate the same blurhash as Civit here: https://github.com/civitai/civitai/blob/8ebc27eadad9feebd7c2e59ceed62544ec34817c/src/utils/blurhash.ts#L75
                                    #
                                    # h = blurhashFile(imageFile)
                                    # s = image['hash']
                                    # if s != h:
                                    #     print(f"🛑 Hash {s} expected but got {h}.  Still...")
                                    # else:
                                    #     print(f"✅ Hash good.  Saving to {hashFileName}")
                                    #     with open(hashFileName, "w") as file:
                                    #         # Write the hash to the file
                                    #         file.write(h)
                                else:
                                    print(f"{lockFileName} detected.  Skipping download.")
                        
                        if not is_file_older_than(imageFile, 24):
                            if S3_BUCKET != None:
                                if not exists_s3(bucket_name = S3_BUCKET, file_name = imageFile) or S3_OVERWRITE:
                                    print("🌎 Uploading to S3...")
                                    upload_file_s3(imageFile, S3_BUCKET, object_name=imageFile, extra_args={"ContentType": "image/jpeg"})
                                else:
                                    print("✅ File already in S3")
                            # else:
                            #     print("No S3 bucket configured")


    except:
        tb = traceback.format_exc()
        print(f"Could not save {model['id']}\n{tb}")