from PIL import Image, ImageFilter, ImageOps
import re
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

from modules.paths import script_path

from modules import devices, sd_samplers
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

from modules.sd_samplers import all_samplers

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

AGENTVERSION = "a1111-v1"
dir_repos = "repositories"
python = sys.executable
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")

# From https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/api/api.py
sampler_to_index = lambda name: next(filter(lambda row: name.lower() == row[1].name.lower(), enumerate(all_samplers)), None)

def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def run(command, desc=None, errdesc=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def repo_dir(name):
    return os.path.join(dir_repos, name)


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run(f'"{git}" -C {dir} rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}").strip()
        if current_hash == commithash:
            return

        run(f'"{git}" -C {dir} fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run(f'"{git}" -C {dir} checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C {dir} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

        
def version_check(commit):
    try:
        import requests
        commits = requests.get('https://api.github.com/repos/AUTOMATIC1111/stable-diffusion-webui/branches/master').json()
        if commit != "<none>" and commits['commit']['sha'] != commit:
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits['commit']['sha'] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:
        print("versipm check failed",e)

        
def prepare_environment():
    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")
    deepdanbooru_package = os.environ.get('DEEPDANBOORU_PACKAGE', "git+https://github.com/KichangKim/DeepDanbooru.git@edf73df4cdaeea2cf00e9ac08bd8a9026b7a7b26")

    xformers_windows_package = os.environ.get('XFORMERS_WINDOWS_PACKAGE', 'https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl')

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/CompVis/stable-diffusion.git")
    taming_transformers_repo = os.environ.get('TAMING_REANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMET_REPO', 'https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')

    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")
    codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
    blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

    sys.argv += shlex.split(commandline_args)

    sys.argv, skip_torch_cuda_test = extract_arg(sys.argv, '--skip-torch-cuda-test')
    sys.argv, reinstall_xformers = extract_arg(sys.argv, '--reinstall-xformers')
    sys.argv, update_check = extract_arg(sys.argv, '--update-check')
    xformers = '--xformers' in sys.argv
    deepdanbooru = '--deepdanbooru' in sys.argv
    ngrok = '--ngrok' in sys.argv

    try:
        commit = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        commit = "<none>"

    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")
    
    if not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

    if not skip_torch_cuda_test:
        run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

    if not is_installed("gfpgan"):
        run_pip(f"install {gfpgan_package}", "gfpgan")

    if not is_installed("clip"):
        run_pip(f"install {clip_package}", "clip")

    if (not is_installed("xformers") or reinstall_xformers) and xformers and platform.python_version().startswith("3.10"):
        if platform.system() == "Windows":
            run_pip(f"install -U -I --no-deps {xformers_windows_package}", "xformers")
        elif platform.system() == "Linux":
            run_pip("install xformers", "xformers")

    if not is_installed("deepdanbooru") and deepdanbooru:
        run_pip(f"install {deepdanbooru_package}#egg=deepdanbooru[tensorflow] tensorflow==2.10.0 tensorflow-io==0.27.0", "deepdanbooru")

    if not is_installed("pyngrok") and ngrok:
        run_pip("install pyngrok", "ngrok")

    os.makedirs(dir_repos, exist_ok=True)

    git_clone(stable_diffusion_repo, repo_dir('stable-diffusion'), "Stable Diffusion", stable_diffusion_commit_hash)
    git_clone(taming_transformers_repo, repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)
    git_clone(k_diffusion_repo, repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
    git_clone(codeformer_repo, repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
    git_clone(blip_repo, repo_dir('BLIP'), "BLIP", blip_commit_hash)

    if not is_installed("lpips"):
        run_pip(f"install -r {os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}", "requirements for CodeFormer")

    run_pip(f"install -r {requirements_file}", "requirements for Worker API")

    if update_check:
        version_check(commit)
    
    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f

def deliver(args, details, duration):
    sample_path = os.path.join(args['out'], f"{details['uuid']}.png")
    url = f"{args['api']}/v3/deliverorder"
    logger.info(url)
    
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
    # Grab start timestamp
    start_time = time.time()
    
    # DEFAULTS
    args = {
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
        "img2img" : False,
        "img2img_denoising_strength" : 0.75,
        "img2img_source_uuid" : "UNKNOWN",
    }

    params = details["params"]
    
    # sampler_index = sampler_to_index(txt2imgreq.sampler_index)

    if "width_height" in params:
        args["width"] = params["width_height"][0]
        args["height"] = params["width_height"][1]

    for param in ["prompt","negative_prompt","scale","steps","seed","denoising_strength","restore_faces","tiling","enable_hr",
        "firstphase_width","firstphase_height","img2img","img2img_denoising_strength","img2img_source_uuid"]:
        if param in params:
            args[param] = params[param]
    
    positive_embeddings = list(re.findall(r"\<(.*?)\>", args["prompt"]))
    negative_embeddings = list(re.findall(r"\<(.*?)\>", args["negative_prompt"]))

    embeddings = list(positive_embeddings + negative_embeddings)
    logger.info(embeddings)
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
        # Reload embeddings
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
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

            p = StableDiffusionProcessingImg2Img(
                init_images=[image],    # img2img PIL Image
                sd_model=shared.sd_model,
                outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                prompt=prompt,
                styles=["None", "None"],
                negative_prompt=negative_prompt,
                seed=args["seed"],
                subseed=0,
                subseed_strength=0,
                seed_resize_from_h=0,
                seed_resize_from_w=0,
                seed_enable_extras=False,
                sampler_index=args["sampler_index"],
                batch_size=1,
                n_iter=args["n_iter"],
                steps=args["steps"],
                cfg_scale=args["scale"],
                width=args["width"],
                height=args["height"],
                restore_faces=args["restore_faces"],
                tiling=args["tiling"],
                denoising_strength=args["img2img_denoising_strength"]
                # tiling=tiling,
                # mask=mask,
                # mask_blur=mask_blur,
                # inpainting_fill=inpainting_fill,
                # resize_mode=resize_mode,
                # inpaint_full_res=inpaint_full_res,
                # inpaint_full_res_padding=inpaint_full_res_padding,
                # inpainting_mask_invert=inpainting_mask_invert,
            )
            processed = modules.scripts.scripts_img2img.run(p, 0)
            if processed is None:
                processed = process_images(p)
                # Grab end timestamp
                end_time = time.time()
                # Capture duration
                duration = end_time - start_time
                print(processed.info)
                generation_info_js = processed.js()
                print(generation_info_js)

                for i, image in enumerate(processed.images):
                    sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
                    image.save(sample_path)
                    deliver(cliargs, details, duration)

                shared.total_tqdm.clear()
        else:
            p = StableDiffusionProcessingTxt2Img(
                do_not_save_samples = True,         # I save them elsewhere
                do_not_save_grid = True,
                sd_model=shared.sd_model,
                outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                # prompt=["photograph","banana"],   # This will override n_iter
                prompt=args["prompt"],
                styles=["None", "None"],
                negative_prompt=args["negative_prompt"],
                seed=args["seed"],
                subseed=0,
                subseed_strength=0,
                seed_resize_from_h=0,
                seed_resize_from_w=0,
                seed_enable_extras=False,
                sampler_index=args["sampler_index"],
                batch_size=1,
                n_iter=args["n_iter"],
                steps=args["steps"],
                cfg_scale=args["scale"],
                width=args["width"],
                height=args["height"],
                restore_faces=args["restore_faces"],
                tiling=args["tiling"],
                enable_hr=args["enable_hr"],
                denoising_strength=args["denoising_strength"] if args["enable_hr"] else None,
                firstphase_width=args["firstphase_width"] if args["enable_hr"] else None,
                firstphase_height=args["firstphase_height"] if args["enable_hr"] else None,
            )
            processed = modules.scripts.scripts_txt2img.run(p, 0)
            if processed is None:
                processed = process_images(p)
                # Grab end timestamp
                end_time = time.time()
                # Capture duration
                duration = end_time - start_time
                print(processed.info)
                generation_info_js = processed.js()
                print(generation_info_js)

                for i, image in enumerate(processed.images):
                    sample_path = os.path.join(cliargs['out'], f"{details['uuid']}.png")
                    image.save(sample_path)
                    deliver(cliargs, details, duration)

                shared.total_tqdm.clear()
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
    # Prepare environment and load model
    logger.info("🔨 Preparing environment...")
    prepare_environment()
    from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
    from modules.shared import opts, cmd_opts

    # Load models
    logger.info("💾 Loading models...")
    queue_lock = threading.Lock()

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()
    modules.scripts.load_scripts()
    shared.sd_model = modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    # Start bot loop
    logger.info("➰ Starting loop...")
    loop(cliargs)

