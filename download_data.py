import os
import requests

repo_url = 'https://github.com/therneau/survival'
folder_path = 'data'
api_url = f'https://api.github.com/repos/{repo_url.split("/")[-2]}/{repo_url.split("/")[-1]}/contents/{folder_path}'

response = requests.get(api_url)
if response.status_code == 200:
    files = response.json()
    os.makedirs(folder_path, exist_ok=True)
    for file in files:
        download_url = file['download_url']
        file_name = os.path.join(folder_path, 'raw', file['name'])
        file_response = requests.get(download_url)
        with open(file_name, 'wb') as f:
            f.write(file_response.content)

    print(f'Downloaded {len(files)} files from the "{folder_path}" folder.')
else:
    print(f'Failed to retrieve data from GitHub. Status code: {response.status_code}')
