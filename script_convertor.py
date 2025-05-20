import yaml

# Load the YAML file
with open("my_jupyter_env.yml", "r") as file:
    env = yaml.safe_load(file)

# Extract dependencies
dependencies = env.get("dependencies", [])

# Write to requirements.txt
with open("requirements.txt", "w") as req_file:
    for dep in dependencies:
        if isinstance(dep, str):  # Ensure it's a valid package name
            req_file.write(dep + "\n")

print("✅ requirements.txt has been created. Install using: pip install -r requirements.txt")
