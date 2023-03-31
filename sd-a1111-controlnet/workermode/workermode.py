from PIL import Image, ImageFilter, ImageOps, PngImagePlugin
import re
import base64
from io import BytesIO
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
import xmlrpc.client

AGENTVERSION = "a1111-v2-controlnet"
CONTROLNET_COMMIT = os.environ.get('CONTROLNET_COMMIT', "UNKNOWN")
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



def deliver(args, details, duration, log):
    sample_path = os.path.join(args['out'], f"{details['uuid']}.png")
    ref_path = os.path.join(args['out'], f"{details['uuid']}_ref.png")
    url = f"{args['api']}/v3/deliverorder"
    
    files = {
        "file": open(sample_path, "rb")
    }
    if os.path.exists(ref_path):
        files["ref_file"] = open(ref_path, "rb")

    logger.info(log)
    values = {
        "duration" : duration,
        "agent_id" : args['agent'],
        "algo" : "stable",
        "repo" : "a1111",
        "agent_version" : AGENTVERSION,
        "owner" : args['owner'],
        "uuid" : details['uuid'],
        "log" : log
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
            if os.path.exists(ref_path):
                os.unlink(ref_path)
                
        except:
            logger.error(f"Error when trying to clean up files for {details['uuid']}")
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)

def do_job(cliargs, details):    
    # DEFAULTS
    args = {
        "sd_model_checkpoint" : "v1-5-pruned-emaonly.ckpt",
        "other_model" : {},
        "parent_uuid" : "UNKNOWN",
        "tiling" : False,
        "firstphase_width" : 0,
        "firstphase_height" : 0,
        "prompt" : "A lonely robot",
        "negative_prompt" : "",
        "seed" : 0,
        "sampler" : "Euler a",
        "n_iter" : 1,
        "steps" : 20,
        "scale" : 7.0,
        "offset_noise" : 0.0,
        "width" : 512,
        "height" : 512,
        "clip_skip" : 1,
        # Face Restore Options
        "restore_faces" : False,
        "fr_model" : "CodeFormer",
        "cf_weight" : 0.5,
        # Highres Options
        "enable_hr" : False,
        "denoising_strength" : 0.75,
        "hr_scale" : 2,
        "hr_upscale" : "None",
        # img2img
        "img2img" : False,
        "img2img_denoising_strength" : 0.75,
        "img2img_source_uuid" : "UNKNOWN",
        # controlnet
        "controlnet_enabled" : False,
        "controlnet_ref_img_type" : "piece",
        "controlnet_guessmode" : False,
        "controlnet_module": "canny",
        "controlnet_model": "control_sd15_canny [fef5e48e]",
        "controlnet_weight": 1,
        "controlnet_guidance": 1,
        # TODO?:
        "controlnet_input_image": [],       # Populated later
        "controlnet_mask": [],
        "controlnet_resize_mode": "Scale to Fit (Inner Fit)",
        "controlnet_lowvram": False,
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 100,
        "controlnet_threshold_b": 200,
        "alwayson_scripts" : {}
    }

    params = details["params"]

    if "width_height" in params:
        args["width"] = params["width_height"][0]
        args["height"] = params["width_height"][1]

    for param in [
        "sd_model_checkpoint","other_model","clip_skip","fr_model","cf_weight",
        "parent_uuid","prompt","negative_prompt","scale","offset_noise","steps","seed","restore_faces","tiling","sampler",
        # Upscale options
        "enable_hr","denoising_strength","hr_scale","hr_upscale",
        # ControlNet options
        "controlnet_enabled","controlnet_ref_img_type","controlnet_ref_img_url","controlnet_guessmode","controlnet_module",
        "controlnet_model","controlnet_weight","controlnet_guidance","controlnet_resizemode",
        "firstphase_width","firstphase_height",
        # img2img options
        "img2img","img2img_denoising_strength","img2img_source_uuid",
    ]:
        # If parameter is in jobparams, override args default.
        if param in params:
            args[param] = params[param]
    
    # Fix old name to new one
    if args["sampler"] == "k_euler_ancestral":
        args["sampler"] = "Euler a"
    
    positive_embeddings = list(re.findall(r"\<(.*?)\>", args["prompt"]))
    negative_embeddings = list(re.findall(r"\<(.*?)\>", args["negative_prompt"]))

    embeddings = list(positive_embeddings + negative_embeddings)
    # logger.info(embeddings)
    
    
    # TODO: Find a new way - Maybe <ti:blabla>?
    # files to import
    # for embedding in embeddings:
    #     try:
    #         # prompt_dict[token] = open(f"{prompt_salad_path}/{token}.txt").read().splitlines()
    #         logger.info(f"⚖️ {embedding} detected.")
    #         embUrl = f"https://www.feverdreams.app/embeddings/{embedding}.pt"
    #         embPath = os.path.join("/home/stable/stable-diffusion-webui/embeddings",f"~~{embedding}~~.pt")
    #         # Download embedding if required
    #         if not os.path.exists(embPath):
    #             logger.info(f"🌍 Downloading {embUrl} to {embPath}...")
    #             response = requests.get(embUrl)
    #             open(embPath, "wb").write(response.content)
    #         else:
    #             logger.info(f"{embPath} found in embeddings dir")
    #     except Exception as e:
    #         # prompt_dict[token] = None
    #         logger.error(f"🛑 Embedding {embedding} could not be found.")
    #         tb = traceback.format_exc()
    #         logger.error(f"{e}\n\n{tb}")
    try:
        prompt = args["prompt"]
        if "offset_noise" in args:
            if args["offset_noise"] > 0.0 or args["offset_noise"] < 0.0:
                prompt = f"{prompt}, <lora:epiNoiseoffset_v2:{args['offset_noise']}>"
    #     prompt = prompt.replace("<","~~")
    #     prompt = prompt.replace(">","~~")
    # 
        negative_prompt = args["negative_prompt"]
    #     negative_prompt = negative_prompt.replace("<","~~")
    #     negative_prompt = negative_prompt.replace(">","~~")

        logger.info(f"ℹ️ Prompt: {prompt}")
        logger.info(f"ℹ️ Negative Prompt: {negative_prompt}")

        # Clear supervisord log
        server = xmlrpc.client.ServerProxy('http://localhost:9001/RPC2')
        logger.info("☠️ Clearing a1111 worker log")
        # Clear the program log
        server.supervisor.clearProcessLogs("auto1111")

        # Load SD model
        if args['sd_model_checkpoint'] == "other":
            om = args['other_model']
            logger.info("🤖 Custom model detected.")
            logger.info(om)
            cdnurl = f"{cliargs['model_cdn']}/models/{om['model_id']}/{om['model_version']}/{om['model_file']}/{om['filename']}"
            cdndir = f"/home/stable/stable-diffusion-webui/models/Stable-diffusion/civitai-cache"
            os.makedirs(cdndir, exist_ok=True)
            filetarget = f"{cdndir}/{om['SHA256'].lower()}.{om['filename'].split('.')[-1]}"
            lockFileName = f"{filetarget}.lock"
            
            # Take no action until lock file is gone
            while os.path.isfile(lockFileName):
                logger.info(f"Looks like the file is being downloaded.  Waiting for 5 seconds...")
                time.sleep(5)
            
            if not os.path.isfile(filetarget):
                if not os.path.isfile(lockFileName):
                    with open(lockFileName, "w") as lockfile:
                        lockfile.write("")
                    try:
                        logger.info(f"Downloading {filetarget} from {cdnurl}...")
                        model = requests.get(cdnurl)
                        response = requests.get(cdnurl)
                        open(filetarget, "wb").write(response.content)
                        logger.info(f"File downloaded.  Refreshing checkpoints in A1111...")
                        results = requests.post(
                        "http://localhost:7860/sdapi/v1/refresh-checkpoints",
                            headers = {'accept': 'application/json', 'Content-Type': 'application/json'},
                            json={"sd_model_checkpoint":args["sd_model_checkpoint"]}
                        ).json()
                    except:
                        pass

                    if os.path.exists(lockFileName):
                        os.remove(lockFileName)
            
            args['sd_model_checkpoint'] = f"civitai-cache_{om['SHA256'].lower()}"
        
        a1111_config = {
            "sd_model_checkpoint" : args["sd_model_checkpoint"],
            "multiple_tqdm" : False
        }

        # if "clip_skip" in args:
        #     a1111_config["CLIP_stop_at_last_layers"] = args['clip_skip']

        if "fr_model" in args:
            a1111_config['face_restoration_model'] = args['fr_model']

        if "cf_weight" in args:
            a1111_config['code_former_weight'] = args['cf_weight']

        logger.info(f"🤖 Setting config {a1111_config}")
        results = requests.post(
            "http://localhost:7860/sdapi/v1/options",
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'},
            json=a1111_config
        ).json()

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
            
            # Retrieve the entire program log content
            server = xmlrpc.client.ServerProxy('http://localhost:9001/RPC2')
            log = server.supervisor.readProcessStdoutLog("auto1111", 0, 0)
            
            image = Image.open(initPath)
            
            # os.unlink(initPath) # Maybe an agent pref later or a cleanup job.

            for i, image in enumerate(processed.images):
                sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
                image.save(sample_path)
                deliver(cliargs, details, duration, log)

        
        logger.info(f"🔮 txt2img Job: \n{args}")
        payload = {
            # Highres
            "enable_hr": args["enable_hr"],
            "denoising_strength": args["denoising_strength"],
            "hr_scale": args["hr_scale"],
            "hr_upscaler": args["hr_upscale"],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "styles": [],
            "seed": args["seed"],
            "subseed": 0,
            "subseed_strength": 0,
            "batch_size": 1,
            "n_iter": args["n_iter"],
            "steps": args["steps"],
            "cfg_scale": args["scale"],
            "width": args["width"],
            "height": args["height"],
            "restore_faces": args["restore_faces"],
            "tiling": args["tiling"],
            "sampler_index": args["sampler"],
            "sampler_name": args["sampler"],
            # TODO: unknown
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": True
            # "script_args": [],
            # "script_name": "",
        }
        
        if args["controlnet_enabled"] == True:
            # TODO: Allow uploaded images
            if args["controlnet_ref_img_type"] == "piece":
                imgurl = f"https://images.feverdreams.app/images/{args['parent_uuid']}.png"
            if args["controlnet_ref_img_type"] == "url":
                imgurl = args["controlnet_ref_img_url"]

            logger.info(f"🌍 Downloading image for ControlNet: {imgurl}")
            b64=url2base64(imgurl)

            lowvram = False
            gpu = list(nvsmi.get_gpus())[0]
            gpu_mem = gpu.__dict__["mem_total"]
            logger.info(f"💻 GPU Memory is {gpu_mem}MB")
            
            # TODO: Decide if I want to do this below for 8GB and under cards.
            # if gpu_mem < 20000:
            #     lowvram = True

            payload["alwayson_scripts"] = {
                "controlnet" : {
                    "args" : [
                        {
                            "input_image": b64,
                            "mask": "",
                            "module": args["controlnet_module"],
                            "model": args["controlnet_model"],
                            "weight": args["controlnet_weight"],
                            "resize_mode": "Scale to Fit (Inner Fit)",
                            "lowvram": lowvram,
                            "processor_res": 512,
                            "threshold_a": 100,
                            "threshold_b": 200,
                            "guidance": args["controlnet_guidance"],
                            "guidance_start": 0,
                            "guidance_end": 1,
                            "guessmode": args["controlnet_guessmode"]
                        }
                    ]
                }
            }
            logger.info(f"🔮 ControlNet Enabled: \n{args}")

        # logger.info(f"🔮 Sending payload to A1111 API...\n{payload}")
        # Grab start timestamp
        start_time = time.time()
        
        results = requests.post(
            "http://localhost:7860/sdapi/v1/txt2img",
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'},
            json=payload
        ).json()
        end_time = time.time()
        # Retrieve the entire program log content
        server = xmlrpc.client.ServerProxy('http://localhost:9001/RPC2')
        log = server.supervisor.readProcessStdoutLog("auto1111", 0, 0)

        # logger.info(results)
        images = results["images"]
        duration = end_time - start_time
        logger.info(f"{len(images)} images returned.")
        image=images[0]
        sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
        image_bytes = base64.b64decode(image)
        image = Image.open(BytesIO(image_bytes))
        # Remove all EXIF and other metadata
        data = list(image.getdata())
        image_without_metadata = Image.new(image.mode, image.size)
        image_without_metadata.putdata(data)
        # Save the stripped image
        image_without_metadata.save(sample_path)
        print('File written successfully.')

        if len(images) > 1:
            ref_image = images[1]
            reference_image_path = os.path.join(cliargs['out'], f"{details['uuid']}_ref.png")
            image_bytes = base64.b64decode(ref_image)
            image = Image.open(BytesIO(image_bytes))
            # Remove all EXIF and other metadata
            data = list(image.getdata())
            image_without_metadata = Image.new(image.mode, image.size)
            image_without_metadata.putdata(data)
            # Save the stripped image
            image_without_metadata.save(reference_image_path)
            print('Reference image written successfully.')

        deliver(cliargs, details, duration, log)

    except Exception as e:
        connected = True #TODO: what?
        if connected:
            tb = traceback.format_exc()
            logger.error(f"Bad job detected.\n\n{e}\n\n{tb}")
            # Retrieve the entire program log content
            server = xmlrpc.client.ServerProxy('http://localhost:9001/RPC2')
            log = server.supervisor.readProcessStdoutLog("auto1111", 0, 0)
            errlog = server.supervisor.readProcessStderrLog("auto1111", 0, 0)
            values = {"message": f"Job failed:\n\n{e}", "traceback": tb, "log" : log, "errlog": errlog}
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
            logger.info("🧪 Checking A1111 API...")
            results = requests.get(
                "http://localhost:7860/sdapi/v1/memory",
                headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            ).json()
            logger.info(results)
            
            # Check API
            try:
                logger.debug(f"🌎 Checking '{url}'...")
                results = requests.post(
                    url,
                    data={
                        "bot_version" : AGENTVERSION,
                        "controlnet_commit" : CONTROLNET_COMMIT,
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
                logger.info("📡 Cannot reach FD API.")
                tb = traceback.format_exc()
                logger.error(tb)
                pass

        except requests.exceptions.ConnectionError as e:
            # tb = traceback.format_exc()
            # logger.error(tb)
            logger.info("📡 Cannot reach A1111 API.  Maybe it is just starting or crashed?")
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
        "model_cdn" : os.environ.get('MODEL_CDN', ""),
        "out" : "/outdir",
        "poll_interval" : 5
    }
    
    # Start bot loop
    logger.info("➰ Starting loop...")
    loop(cliargs)

