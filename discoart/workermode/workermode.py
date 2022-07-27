import shutil
from loguru import logger
import time, requests, json, os
from discoart import create
from discoart import __version__
from dotenv import load_dotenv
from docarray import Document
from docarray import DocumentArray
import traceback

# Function definitions
def parse_seed(details):
    import random
    random.seed()
    seed = random.randint(0, 2**32)
    
    if "set_seed" in details:
        if details["set_seed"] != -1 and details["set_seed"] != "random_seed":
            seed = details["set_seed"]
            logger.info(f"🌱Specific seed in job selected.  Using {seed} in discoart...")   
    return seed
            
def parse_text_prompts(details):
    if "text_prompts" in details:
        return details["text_prompts"]
    # Legacy
    if "text_prompt" in details:
        tp = details["text_prompt"]

    # Attempt to accept JSON Structured Text Prompt...
    if tp:
        try:
            tp = eval(tp)
            if type(tp) == list:
                logger.info("JSON structured text prompt found.")
                return tp
            else:
                raise Exception("Non-list item found")
        except:
            tp = details["text_prompt"]
            tp = tp.replace(":", "")
            tp = tp.replace('"', "")
            logger.debug("Flat string text prompt found.")
            return [tp]
    
    return None

def parse_width_height(details):
    w_h = [1024, 768]
    if "width_height" in details:
        logger.info("Width/Height detected")
        return details["width_height"]
    if type(details["shape"]) == str:
        logger.info("Legacy Shape string detected")
        shapes = {
            "square": [1024, 1024],
            "tiny_square": [512, 512],
            "landscape": [1024, 1024],
            "portait": [1024, 1024],
            "pano": [2048, 512],
            "skyscraper": [512, 2048],
        }
        w_h = shapes.get(details["shape"], [1024,1024])
    
    return w_h

def do_job(args, details):
    docname = f"discoart-{details['uuid']}"
    text_prompts = parse_text_prompts(details)
    width_height = parse_width_height(details)
    diffusion_model = details["diffusion_model"]
    seed = parse_seed(details)

    import os, time

    ### Disable log spam
    os.environ["DISCOART_DISABLE_RESULT_SUMMARY"] = "1"
    os.environ["DISCOART_DISABLE_IPYTHON"] = '1'

    # Grab start timestamp
    start_time = time.time()

    # Run Disco
    try:
        da = create(
            # FD-hardcoded
            name_docarray = docname,
            n_batches = 1,
            batch_size = 1,
            display_rate = 20,
            truncate_overlength_prompt = True,
            gif_fps = 0,
            # User params
            text_prompts = text_prompts,
            seed = seed,
            steps=details['steps'],
            skip_steps = 0,
            width_height = width_height,
            diffusion_model = details["diffusion_model"],
            use_secondary_model = details["use_secondary_model"],
            diffusion_sampling_mode = "ddim",
            clip_models = details["clip_models"],
            cutn_batches = details["cutn_batches"],
            cut_overview = details["cut_overview"], 
            cut_ic_pow = details["cut_ic_pow"],
            cut_innercut = details["cut_innercut"],
            cut_icgray_p = details["cut_icgray_p"],
            range_scale = details["range_scale"],
            sat_scale = details["sat_scale"],
            clamp_grad = details["clamp_grad"],
            clamp_max = details["clamp_max"],
            eta = details["eta"],
            use_horizontal_symmetry = details["use_horizontal_symmetry"],
            use_vertical_symmetry = details["use_vertical_symmetry"],
            transformation_percent = details["transformation_percent"],
            randomize_class = details["randomize_class"],
            skip_augs = details["skip_augs"],
            clip_denoised = details["clip_denoised"],
            fuzzy_prompt = details["fuzzy_prompt"],
            # diffusion_model_config = None,
            # init_scale = 1000.0,
            perlin_init = False,
            rand_mag = 0.05,
        )

        # Grab end timestamp
        end_time = time.time()

        # Capture duration
        duration = end_time - start_time

        post_process(da, details)
        deliver(args, da, details, duration)
    except Exception as e:
        connected = True #TODO: what?
        if connected:
            tb = traceback.format_exc()
            logger.error(f"Bad job detected.\n\n{e}\n\n{tb}")
            values = {"message": f"Job failed:\n\n{e}", "traceback": tb}
            requests.post(f"{args.dd_api}/reject/{args.agent}/{details['uuid']}", data=values)
        else:
            logger.error(f"Error.  Check your API host is running at that location.  Also check your own internet connectivity.  Exception:\n{tb}")
            raise(tb)

## Post-processing
def post_process(da, details):
    
    # Save sprite sheet
    sprites = f"{details['uuid']}_sprites.png"
    da[0].chunks.plot_image_sprites(
        output = sprites,
        skip_empty=True,
        show_index=False,
        keep_aspect_ratio=True
    )

    # Save progress frames as GIF
    gif = f"{details['uuid']}.gif"
    da[0].chunks.save_gif(
        gif,
        show_index=False,
        inline_display=False,
        size_ratio=0.5
    )

    document_name = f"{details['uuid']}.protobuf.lz4"

    # Strip out progress images to save bandwidth
    da[0].chunks = []
    da[0].chunks.append(Document(uri=gif, tags = {"name" : "render_animation"}))
    da[0].chunks.append(Document(uri=sprites, tags = {"name" : "sprite_sheet"}))
    da.save_binary(document_name)

def deliver(args, da, details, duration):
    url = f"{args.dd_api}/v2/deliverorder"
    document_name = f"{details['uuid']}.protobuf.lz4"
    # gif_name = f"{details['uuid']}.gif"
    # sprite_name = f"{details['uuid']}_sprites.png"
    files = {
        "file": open(document_name, "rb"),
        # "gif" : open(gif_name, "rb"),
        # "sprite" : open(sprite_name, "rb"),
    }
    values = {
        "duration" : duration,
        "agent_id" : args.agent,
        "agent_version" : "3.0",
        "owner" : args.owner,
        "uuid" : details['uuid'],
        "agent_discoart_version" : __version__,
        "agent_build_version" : os.getenv("DISCOART_VERSION")
    }
    # Upload payload
    try:
        logger.info(f"🌍 Uploading {document_name} to {url}...")
        results = requests.post(url, files=files, data=values)
        feedback = results.json()
        logger.info(feedback)
        # Clean up
        os.unlink(f"{document_name}")
    except:
        logger.error("Error uploading LZ4.")
      
    if not os.getenv('WORKER_SAVEFILES'):
        try:
            # Clean up files
            os.unlink(f"{document_name}")
            # os.unlink(f"{gif_name}")
            # os.unlink(f"{sprite_name}")
            # Clean up LZ4s
            os.unlink(f"{os.getenv('DISCOART_OUTPUT_DIR')}/discoart-{details['uuid']}.protobuf.lz4")
            os.unlink(f"{os.getenv('DISCOART_OUTPUT_DIR')}/{details['uuid']}.protobuf.lz4")
            # Clean up directory
            shutil.rmtree(f"DISCOART_OUTPUT_DIR/discoart-{details['uuid']}") 
        except:
            logger.error(f"Error when trying to clean up files for {details['uuid']}")

def loop(args):
    # Start bot loop
    run = True
    idle_time = 0
    start_time = time.time()
    
    DD_AGENTVERSION = "3.0"

    while run:
        url = f"{args.dd_api}/v2/takeorder/{args.agent}"
        try:
            logger.debug(f"🌎 Checking '{url}'...")
            results = requests.post(
                url,
                data={
                    "bot_version": DD_AGENTVERSION,
                    "owner": args.owner,
                    "idle_time": idle_time,
                    "model": "custom",
                    "agent_discoart_version" : __version__,
                    "agent_build_version" : os.getenv("DISCOART_VERSION"),
                    "start_time" : start_time
                }
            ).json()
            if results["success"]:
                details = results["details"]
                logger.info(f"Job {details['uuid']} received.")
                idle_time = 0
                do_job(args, details)
            else:
                logger.error(results)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            pass
        
        logger.info(f"Sleeping for {args.poll_interval} seconds...  I've been sleeping for {idle_time} seconds.")
        time.sleep(args.poll_interval)
        idle_time = idle_time + args.poll_interval

def main():
    # Environment-specific settings:
    load_dotenv()
    import argparse
    parser = argparse.ArgumentParser(description="Disco Diffusion Worker")
    parser.add_argument("--dd_api", help="Disco worker API http endpoint", required=True, default=os.getenv("DD_API"))
    parser.add_argument("--agent", help="Disco worker agent name", required=True, default=os.getenv("DD_AGENTNAME"))
    parser.add_argument("--owner", help="Disco worker agent owner", required=False, default=398901736649261056)
    parser.add_argument("--images_out", help="Directory for render jobs", required=False, default=os.getenv("DD_IMAGES_OUT", "/workspace/out"))
    # parser.add_argument("--cuda_device", help="CUDA Device", required=False, default=os.getenv("DD_CUDA_DEVICE", "cuda:0"))
    parser.add_argument("--poll_interval", type=int, help="Polling interval between jobs", required=False, default=os.getenv("DD_POLLINTERVAL", 5))
    parser.add_argument("--dream_time", type=int, help="Time in seconds until dreams", required=False, default=os.getenv("DD_POLLINTERVAL", 300))
    args = parser.parse_args()
    loop(args)


if __name__ == "__main__":
    main()
