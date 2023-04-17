import json

# Currently (Feb 2023), the /sdapi/v1/options POST method is unreliable and generates 500 server errors
# This is a pre-start workaround

# Load the config file
try:
    with open("/home/stable/stable-diffusion-webui/config.json", "r") as f:
        config = json.load(f)
except:
    config = {}

# Set the desired values
config["control_net_allow_script_control"] = True
config["control_net_models_path"] = "/home/stable/stable-diffusion-webui/models/cn"
config["sd_model_checkpoint"] = "v1-5-pruned-emaonly.ckpt"
config["sd_checkpoint_hash"] = "cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516"
config["samples_save"] = False
config["grid_save"] = False
config["do_not_add_watermark"] = True
config["enable_pnginfo"] = False
config["show_progressbar"] = False

# Save the modified config file
with open("/home/stable/stable-diffusion-webui/config.json", "w") as f:
    json.dump(config, f, indent=4)
