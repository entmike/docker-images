import os, sys
import cv2
import matplotlib.pyplot as plt
import boto3
from PIL import Image
from types import SimpleNamespace
from pymongo import UpdateOne
import boto3
from IPython.display import clear_output
import hashlib
import uuid
import json, random
import argparse, sys, glob
import urllib.request
import requests
from loguru import logger

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

# Environment Vars
from dotenv import load_dotenv
load_dotenv()

WORKER = os.getenv('WORKER')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')
ORIGINAL_URL_PREFIX = os.getenv('ORIGINAL_URL_PREFIX')
CLOUDFLARE_TOKEN = os.getenv('CLOUDFLARE_TOKEN')
CLOUDFLARE_ENDPOINT = os.getenv('CLOUDFLARE_ENDPOINT')

os.makedirs('/out', exist_ok=True)
os.makedirs('/out/face_enhanced', exist_ok=True)
os.makedirs('/out/upscaled', exist_ok=True)
os.makedirs('/out/final', exist_ok=True)

watermark = Image.open("/workspace/watermark.png")
wm_width, wm_height = watermark.size

model_name = 'RealESRGAN_x4plus'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
# determine model paths
model_path = os.path.join('/workspace/Real-ESRGAN/experiments/pretrained_models', model_name + '.pth')
if not os.path.isfile(model_path):
    model_path = os.path.join('/workspace/Real-ESRGAN/realesrgan/weights', model_name + '.pth')
if not os.path.isfile(model_path):
    raise ValueError(f'Model {model_name} does not exist.')


upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
    # tile=params["tile"],
    # tile_pad=params["tile_pad"],
    # pre_pad=params["pre_pad"],
    # half=not params["fp32"],
    gpu_id=0)

face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def get_database():
    from pymongo import MongoClient
    import pymongo

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient

    client = MongoClient(MONGODB_CONNECTION)
    return client

linode_obj_config = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "endpoint_url": ENDPOINT_URL,
}
s3_client = boto3.client("s3", **linode_obj_config)

with get_database() as client:
    operations = []
    records = list(client.stockphotos.stockphotos80.aggregate([
        {
            "$match" : {
                "render_metadata.hash" : {"$exists" : True},
                "$or" : [
                    {
                        "render_metadata.face_enhanced" : {"$exists" : False}
                    },
                    {
                        "render_metadata.face_enhance_worker" : WORKER,
                        "render_metadata.face_enhanced" : False 
                    }
                ]
            }
        },{
            "$limit" : BATCH_SIZE
        }
    ]))

    reserve = True
    if len(records) > 0 and reserve == True:
        # Reserve Records
        for record in records:
            operations.append(
                UpdateOne({'_id': record.get('_id')}, {'$set': {
                    "render_metadata.face_enhance_worker" : WORKER,
                    "render_metadata.face_enhanced" : False
                }})
            )

        result = client.stockphotos.stockphotos80.bulk_write(operations)
           
    for record in records:
        try:
            render_metadata = record["render_metadata"]
            url = f"{ORIGINAL_URL_PREFIX}/{render_metadata['hash']}.png"
            filepath = os.path.join('/out/', f"{render_metadata['hash']}.png")
            logger.info(f"🌍 Downloading {url} to {filepath}...")
            urllib.request.urlretrieve(url, filepath)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            face_save_path = os.path.join('/out/face_enhanced/', f"{render_metadata['hash']}.png")
            logger.info(f"Saving face enhanced image to '{face_save_path}'")      
            cv2.imwrite(face_save_path, output)

            img = cv2.imread(face_save_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=2)
            upscaled_save_path = os.path.join('/out/upscaled/', f"{render_metadata['hash']}.png")
            logger.info(f"Saving upscaled image to '{upscaled_save_path}'")
            cv2.imwrite(upscaled_save_path, output)

            image = Image.open(upscaled_save_path)
            width, height = image.size
            image.paste(watermark, (width - wm_width - 15, height - wm_height - 15), watermark)
            final_save_path = os.path.join('/out/final/', f"{render_metadata['hash']}.png")
            logger.info(f"Saving watermarked image to '{final_save_path}'")
            image.save(final_save_path)
            
            # logger.info(f"Uploading face enhanced image to Linode...")
            # s3_client.upload_file(
            #     Filename = final_save_path,
            #     Bucket = 'fullsize',
            #     Key = f"{render_metadata['hash']}.png",
            #     ExtraArgs = {
            #         "ACL" : "public-read",
            #         "ContentType": "image/png"
            #     }
            # )
            
            logger.info(f"Uploading watermarked image to Linode...")
            s3_client.upload_file(
                Filename = final_save_path,
                Bucket = 'fullsize',
                Key = f"{render_metadata['hash']}.png",
                ExtraArgs = {
                    "ACL" : "public-read",
                    "ContentType": "image/png"
                }
            )

            # files = {'file': open(filepath, 'rb')}
            # headers = {
            #     'Authorization': CLOUDFLARE_TOKEN
            # }
            # logger.info(f"Uploading original image to Cloudflare...")
            # r = requests.post(CLOUDFLARE_ENDPOINT, files=files, headers=headers)
            # cf_metadata_original = r.json()
            
            files = {'file': open(face_save_path, 'rb')}
            headers = {
                'Authorization': CLOUDFLARE_TOKEN
            }
            logger.info(f"Uploading face enhanced image to Cloudflare...")
            r = requests.post(CLOUDFLARE_ENDPOINT, files=files, headers=headers)
            cf_metadata = r.json()
            
            client.stockphotos.stockphotos80.update_one(
                {"render_metadata.hash" : render_metadata['hash']},
                {"$set" : {
                    "render_metadata.face_enhanced" : True,
                    "render_metadata.upscaled" : True,
                    "render_metadata.watermarked" : True,
                    # "cloudflare_metadata_original" : cf_metadata_original["result"],
                    "cloudflare_metadata" : cf_metadata["result"]
                }
            })
            logger.info("🧹 Cleaning up files...")
            os.unlink(filepath)
            os.unlink(face_save_path)
            os.unlink(upscaled_save_path)
            os.unlink(final_save_path)
        except:
            logger.error(f"An error occured")
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            pass
        
        