import os
import requests

repo_url = 'https://github.com/therneau/survival'
folder_path = 'data'

# Optional: If the repository is private or requires authentication, provide your GitHub token.
# token = 'YOUR_GITHUB_TOKEN'

# Construct the API URL to list files in the folder
api_url = f'https://api.github.com/repos/{repo_url.split("/")[-2]}/{repo_url.split("/")[-1]}/contents/{folder_path}'

# Optional: Include authentication headers if using a GitHub token
# headers = {'Authorization': f'token {token}'}

# Make a GET request to the GitHub API
response = requests.get(api_url)  # Use headers=headers if authentication is needed

if response.status_code == 200:
    # Parse the JSON response to get the list of files
    files = response.json()
    
    # Create a directory to store the downloaded files
    os.makedirs(folder_path, exist_ok=True)

    # Loop through the files and download each one
    for file in files:
        # Get the download URL for the file
        download_url = file['download_url']

        # Get the file name
        file_name = os.path.join(folder_path, 'raw', file['name'])

        # Download the file
        file_response = requests.get(download_url)
        with open(file_name, 'wb') as f:
            f.write(file_response.content)

    print(f'Downloaded {len(files)} files from the "{folder_path}" folder.')
else:
    print(f'Failed to retrieve data from GitHub. Status code: {response.status_code}')
