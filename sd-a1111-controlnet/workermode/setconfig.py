import json

# Currently (Feb 2023), the /sdapi/v1/options POST method is unreliable and generates 500 server errors
# This is a pre-start workaround

# Load the config file
try:
    with open("/home/stable/stable-diffusion-webui/config.json", "r") as f:
        config = json.load(f)
except:
    config = {}

# Set the desired value to True
config["control_net_allow_script_control"] = True

# Save the modified config file
with open("/home/stable/stable-diffusion-webui/config.json", "w") as f:
    json.dump(config, f, indent=4)
