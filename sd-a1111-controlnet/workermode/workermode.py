from PIL import Image, ImageFilter, ImageOps
import re
import base64
import argparse
import os, time, requests, json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
import traceback
import nvsmi
import shutil
import psutil
from cgitb import enable
import threading
from difflib import restore
import subprocess
import os
import sys
import importlib.util
import shlex
import platform

AGENTVERSION = "a1111-v2-controlnet"
index_url = os.environ.get('INDEX_URL', "")

def url2base64(url):
    response = requests.get(url)  # download the image from the URL
    if response.status_code == 200:  # check if the request was successful
        image_data = response.content  # get the image content as bytes
        encoded_image = base64.b64encode(image_data)  # base64-encode the image
        b64=encoded_image.decode()
        return b64
    else:
        print('Error: could not download image')  # print an error message if the request failed
        return None



def deliver(args, details, duration):
    sample_path = os.path.join(args['out'], f"{details['uuid']}.png")
    url = f"{args['api']}/v3/deliverorder"
    
    files = {
        "file": open(sample_path, "rb")
    }
    values = {
        "duration" : duration,
        "agent_id" : args['agent'],
        "algo" : "stable",
        "repo" : "a1111",
        "agent_version" : AGENTVERSION,
        "owner" : args['owner'],
        "uuid" : details['uuid']
    }
    # Upload payload
    try:
        logger.info(f"🌍 Uploading {sample_path} to {url}...")
        results = requests.post(url, files=files, data=values)
        try:
            feedback = results.json()
        except:
            feedback = results.text
        logger.info(feedback)
    except:
        logger.error("Error uploading image.")
        tb = traceback.format_exc()
        logger.error(tb)

    if not os.getenv('WORKER_SAVEFILES'):
        try:
            # Delete sample
            os.unlink(sample_path)
        except:
            logger.error(f"Error when trying to clean up files for {details['uuid']}")
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)

def do_job(cliargs, details):    
    # DEFAULTS
    args = {
        "parent_uuid" : "UNKNOWN",
        "enable_hr" : False,
        "tiling" : False,
        "restore_faces" : False,
        "firstphase_width" : 0,
        "firstphase_height" : 0,
        "denoising_strength" : 0.75,
        "prompt" : "A lonely robot",
        "negative_prompt" : "",
        "seed" : 0,
        "sampler_index" : 0,
        "n_iter" : 1,
        "steps" : 20,
        "scale" : 7.0,
        "width" : 512,
        "height" : 512,
        # img2img
        "img2img" : False,
        "img2img_denoising_strength" : 0.75,
        "img2img_source_uuid" : "UNKNOWN",
        # controlnet
        "controlnet_enabled" : False,
        "controlnet_module": "canny",
        "controlnet_model": "control_sd15_canny [fef5e48e]",
        "controlnet_weight": 1,
        "controlnet_guidance": 1,
        # TODO?:
        "controlnet_input_image": [],
        "controlnet_mask": [],
        "controlnet_resize_mode": "Scale to Fit (Inner Fit)",
        "controlnet_lowvram": True,
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 100,
        "controlnet_threshold_b": 200,
    }

    params = details["params"]
    
    # sampler_index = sampler_to_index(txt2imgreq.sampler_index)

    if "width_height" in params:
        args["width"] = params["width_height"][0]
        args["height"] = params["width_height"][1]

    for param in ["parent_uuid","prompt","negative_prompt","scale","steps","seed","denoising_strength","restore_faces","tiling","enable_hr",
        "firstphase_width","firstphase_height","img2img","img2img_denoising_strength","img2img_source_uuid",
        "controlnet_enabled","controlnet_module","controlnet_model","controlnet_weight","controlnet_guidance"]:
        if param in params:
            args[param] = params[param]
    
    positive_embeddings = list(re.findall(r"\<(.*?)\>", args["prompt"]))
    negative_embeddings = list(re.findall(r"\<(.*?)\>", args["negative_prompt"]))

    embeddings = list(positive_embeddings + negative_embeddings)
    # logger.info(embeddings)
    # files to import
    for embedding in embeddings:
        try:
            # prompt_dict[token] = open(f"{prompt_salad_path}/{token}.txt").read().splitlines()
            logger.info(f"⚖️ {embedding} detected.")
            embUrl = f"https://www.feverdreams.app/embeddings/{embedding}.pt"
            embPath = os.path.join("/home/stable/stable-diffusion-webui/embeddings",f"~~{embedding}~~.pt")
            # Download embedding if required
            if not os.path.exists(embPath):
                logger.info(f"🌍 Downloading {embUrl} to {embPath}...")
                response = requests.get(embUrl)
                open(embPath, "wb").write(response.content)
            else:
                logger.info(f"{embPath} found in embeddings dir")
        except Exception as e:
            # prompt_dict[token] = None
            logger.error(f"🛑 Embedding {embedding} could not be found.")
            tb = traceback.format_exc()
            logger.error(f"{e}\n\n{tb}")
    try:
        prompt = args["prompt"]
        prompt = prompt.replace("<","~~")
        prompt = prompt.replace(">","~~")

        negative_prompt = args["negative_prompt"]
        negative_prompt = negative_prompt.replace("<","~~")
        negative_prompt = negative_prompt.replace(">","~~")

        logger.info(f"ℹ️ Prompt: {prompt}")
        logger.info(f"ℹ️ Negative Prompt: {negative_prompt}")

        if(args["img2img"]):

            initUrl = f"https://images.feverdreams.app/images/{args['img2img_source_uuid']}.png"
            initPath = os.path.join("/tmp",f"{args['img2img_source_uuid']}.png")
            # Download init if required
            if not os.path.exists(initPath):
                logger.info(f"🌍 Downloading image {initUrl} to {initPath}...")
                response = requests.get(initUrl)
                open(initPath, "wb").write(response.content)
            else:
                logger.info(f"{initPath} found in tmp dir")
            
            image = Image.open(initPath)
            
            # os.unlink(initPath) # Maybe an agent pref later or a cleanup job.

            for i, image in enumerate(processed.images):
                sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
                image.save(sample_path)
                deliver(cliargs, details, duration)
        else:
            # txt2img API Call here
            # TODO: Allow uploaded/external images
            logger.info(args)
            imgurl = f"https://images.feverdreams.app/images/{args['parent_uuid']}.png"
            logger.info(f"🌍 Downloading image for ControlNet: {imgurl}")
            b64=url2base64(imgurl)
            payload={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "controlnet_input_image": [b64],
                "controlnet_module": args["controlnet_module"],
                "controlnet_model": args["controlnet_model"],
                "controlnet_weight": args["controlnet_weight"],
                "controlnet_guidance": args["controlnet_guidance"],
                "controlnet_resize_mode": "Scale to Fit (Inner Fit)",
                "enable_hr": False,
                "denoising_strength": 0.5,
                "hr_scale": 1.5,
                "hr_upscale": "Latent",
                "guess_mode": False,
                "seed": args["seed"],
                "subseed": 0,
                "subseed_strength": 0,
                "sampler_index": "Euler a",
                "batch_size": 1,
                "n_iter": args["n_iter"],
                "steps": args["steps"],
                "cfg_scale": args["scale"],
                "width":args["width"],
                "height":args["height"],
                "restore_faces": args["restore_faces"],
                # TODO?:
                "controlnet_mask": [],
                "controlnet_lowvram": True,
                "controlnet_processor_res": 512,
                "controlnet_threshold_a": 100,
                "controlnet_threshold_b": 200,
                "override_settings": {},
                "override_settings_restore_afterwards": True
            }
            # logger.info(f"🔮 Sending payload to A1111 API...\n{payload}")
            # Grab start timestamp
            start_time = time.time()
            results = requests.post(
                "http://localhost:7860/controlnet/txt2img",
                headers = {'accept': 'application/json', 'Content-Type': 'application/json'},
                json=payload
            ).json()
            end_time = time.time()

            logger.info(results)
            images = results["images"]
            duration = end_time - start_time
            #for image in images:
            image = images[0]
            sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
            with open(sample_path, 'wb') as f:
                f.write(base64.b64decode(image))
                print('File written successfully.')
            deliver(cliargs, details, duration)

    except Exception as e:
        connected = True #TODO: what?
        if connected:
            tb = traceback.format_exc()
            logger.error(f"Bad job detected.\n\n{e}\n\n{tb}")
            values = {"message": f"Job failed:\n\n{e}", "traceback": tb}
            requests.post(f"{cliargs['api']}/v3/reject/{cliargs['agent']}/{details['uuid']}", data=values)
        else:
            logger.error(f"Error.  Check your API host is running at that location.  Also check your own internet connectivity.  Exception:\n{tb}")
            raise(tb)

def loop(args):
    # Start bot loop
    run = True
    idle_time = 0
    boot_time = datetime.utcnow()

    while run:
        start_time = time.time()
        gpu = list(nvsmi.get_gpus())[0]
        total, used, free = shutil.disk_usage(args["out"])
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
        
        url = f"{args['api']}/v3/takeorder/{args['agent']}"
        try:
            logger.debug(f"🌎 Checking '{url}'...")
            results = requests.post(
                url,
                data={
                    "bot_version": AGENTVERSION,
                    "algo" : "stable",
                    "repo" : "a1111",
                    "gpus": gpu_record,
                    "owner": args["owner"],
                    "idle_time": idle_time,
                    "start_time" : start_time,
                    "free_space" : free,
                    "total_space" : total,
                    "used_space" : used,
                    "boot_time" : boot_time,
                    "memory" : memdict
                }
            ).json()
            
            if "command" in results:
                if results["command"] == 'terminate':
                    logger.info("🛑 Received terminate instruction.  Cya.")
                    run = False    
                    
            if results["success"]:
                if "details" in results:
                    details = results["details"]
                    logger.info(f"Job {details['uuid']} received.")
                    idle_time = 0
                    do_job(args, details)
        
            else:
                logger.error(results)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(tb)
            pass
        
        if run:
            poll_interval = args["poll_interval"]
            poll_interval = 5
            logger.info(f"Sleeping for {poll_interval} seconds...  I've been sleeping for {idle_time} seconds.")
            time.sleep(poll_interval)
            idle_time = idle_time + poll_interval
        else:
            logger.info("Terminating loop.")


if __name__ == "__main__":
    # Environment-specific settings:
    load_dotenv()
    cliargs = {
        "api" : os.environ.get('API_URL', ""),
        "agent" : os.environ.get('API_AGENT', ""),
        "owner" : os.environ.get('API_OWNER', ""),
        "out" : "/outdir",
        "poll_interval" : 5
    }
    
    # Start bot loop
    logger.info("➰ Starting loop...")
    loop(cliargs)

