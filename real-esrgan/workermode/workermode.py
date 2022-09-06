import argparse
import cv2
import glob
import shutil
import nvsmi
import subprocess
import urllib.request
from loguru import logger
import time, requests, json, os
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta
from types import SimpleNamespace

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

load_dotenv()
AGENTVERSION = 'esrgan_v1'

def deliver(args, job, duration):
    url = f"{args.api}/v3/deliveresrgan"
    filepath = os.path.join(args.images_out, f"{job['augid']}.png")
    logger.info(url)
    files = {
        "file": open(filepath, "rb")
    }
    values = {
        "duration" : 0.0,
        "agent_id" : args.agent,
        "algo" : "esrgan",
        "agent_version" : AGENTVERSION,
        "owner" : args.owner,
        "uuid" : job['params']['uuid'],
        "augid" : job['augid'],
    }
    # Upload payload
    try:
        logger.info(f"🌍 Uploading {filepath} to {url}...")
        results = requests.post(url, files=files, data=values)
        try:
            feedback = results.json()
        except:
            feedback = results.text
        logger.info(feedback)
    except:
        logger.error("Error uploading image.")
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)

    # if not os.getenv('WORKER_SAVEFILES'):
    #     try:
    #         # Delete sample
    #         os.unlink(filepath)
    #         # Clean up directory
    #         shutil.rmtree(sample_path)
    #     except:
    #         logger.error(f"Error when trying to clean up files for {details['uuid']}")
    #         import traceback
    #         tb = traceback.format_exc()
    #         logger.error(tb)
            
def do_job(args, job):
    # determine models according to model names
    logger.info(job)
    params = job["params"]
    model_name = params["model_name"]
    
    if not model_name:
        model_name = "RealESRGAN_x4plus"

    model_name = model_name.split('.')[0]
    if model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join('/workspace/Real-ESRGAN/experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('/workspace/Real-ESRGAN/realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    # restorer
    print("Loading restorer")
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=params["tile"],
        tile_pad=params["tile_pad"],
        pre_pad=params["pre_pad"],
        half=not params["fp32"],
        gpu_id=0)
  
    uuid = params["uuid"]
    augid = job["augid"]
    url = f"https://s3.us-east-1.amazonaws.com/images.feverdreams.app/images/{uuid}.png"

    logger.info(f"🌍 Downloading {url}...") 
    filepath = os.path.join(args.images_out, f"{uuid}.png")
    urllib.request.urlretrieve(url, filepath)
    imgname, extension = os.path.splitext(os.path.basename(filepath))
    print(f'Upscaling {imgname}')

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    try:
        if params["face_enhance"]:
            print("Loading face enhancer.")
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=params["outscale"],
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=params["outscale"])
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = extension[1:]
        save_path = os.path.join(args.images_out, f'{augid}.{extension}')
        logger.info(f"Saving augmented image to '{save_path}'")
        cv2.imwrite(save_path, output)
    
    duration = 0.0
    
    deliver(args, job, duration)
    
def loop(args):
    # Start bot loop
    run = True
    idle_time = 0
    boot_time = datetime.utcnow()
    os.makedirs(args.images_out, exist_ok=True)
    while run:
        start_time = time.time()
        gpu = list(nvsmi.get_gpus())[0]
        total, used, free = shutil.disk_usage(args.images_out)
        import psutil
        try:
            meminfo = psutil.virtual_memory()
            memdict = {}
            memdict['total'] = meminfo.total/1024
            memdict['used'] = meminfo.used/1024
            memdict['free'] = meminfo.free/1024
            memdict['buffers'] = meminfo.buffers/1024
            memdict['cached'] = meminfo.cached/1024
            memdict['percent'] = meminfo.percent
        except:
            memdict = {}

        gpu_record = {}
        for key in list(gpu.__dict__.keys()):
            gpu_record[key]=gpu.__dict__[key]
    
        gpu_record = json.dumps(gpu_record)
        memdict = json.dumps(memdict)
        
        url = f"{args.api}/v3/takeorder/{args.agent}"
        
        try:
            logger.debug(f"🌎 Checking '{url}'...")
            results = requests.post(
                url,
                data={
                    "bot_version": AGENTVERSION,
                    "algo" : "esrgan",
                    "gpus": gpu_record,
                    "owner": args.owner,
                    "idle_time": idle_time,
                    "start_time" : start_time,
                    "free_space" : free,
                    "total_space" : total,
                    "used_space" : used,
                    "boot_time" : boot_time,
                    "memory" : memdict
                }
            ).json()
            logger.info(results)
            if "command" in results:
                if results["command"] == 'terminate':
                    logger.info("🛑 Received terminate instruction.  Cya.")
                    run = False    
                    
            if results["success"]:
                if "details" in results:
                    details = results["details"]
                    logger.info(f"Job {details['params']['uuid']} received.")
                    idle_time = 0
                    do_job(args, details)
            else:
                logger.error(results)
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            pass
        
        if run:
            poll_interval = args.poll_interval
            poll_interval = 5
            logger.info(f"Sleeping for {poll_interval} seconds...  I've been sleeping for {idle_time} seconds.")
            time.sleep(poll_interval)
            idle_time = idle_time + poll_interval
        else:
            logger.info("Terminating loop.")

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="Worker API http endpoint", required=True, default=os.getenv("DD_API"))
    parser.add_argument("--agent", help="Worker agent name", required=True, default=os.getenv("AGENTNAME"))
    parser.add_argument("--owner", help="Worker agent owner", required=False, default=398901736649261056)
    parser.add_argument("--images_out", help="Directory for render jobs", required=False, default=os.getenv("DD_IMAGES_OUT", "/workspace/out"))
    parser.add_argument("--poll_interval", type=int, help="Polling interval between jobs", required=False, default=os.getenv("DD_POLLINTERVAL", 0))
    parser.add_argument("--dream_time", type=int, help="Time in seconds until dreams", required=False, default=os.getenv("DD_POLLINTERVAL", 300))   
    
    args = parser.parse_args()
    loop(args)

if __name__ == '__main__':
    main()