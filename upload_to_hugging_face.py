from huggingface_hub import login
import os
from huggingface_hub import upload_file

# Replace 'your_huggingface_token' with your actual token
login(token="your_huggingface_token")

# Define the paths to the folders
directories = ["fra", "jpn"]

# Loop through directories and upload files
for directory in directories:
    folder_path = f"/home/exouser/MMMU/{directory}"
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Define where in the repo the file should go
            repo_file_path = os.path.relpath(file_path, "/home/exouser/MMMU")

            # Upload file to Hugging Face repo
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_file_path,
                repo_id="utischoolnlp/translated_mmmu_datasets",
                repo_type="dataset"
            )
            print(f"Uploaded {file_path} to {repo_file_path}")
