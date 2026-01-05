import pandas as pd
import requests
import os
import shutil
from kaggle_secrets import UserSecretsClient
from tqdm import tqdm
import time

# --- CONFIGURATION ---
user_secrets = UserSecretsClient()
API_KEY = user_secrets.get_secret("MAPBOX_KEY")

OUTPUT_DIR = '/kaggle/working/satellite_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD AND COMBINE DATA ---
train_path = '/kaggle/input/cdcpro/train(1).xlsx'
test_path = '/kaggle/input/cdcpro/test2.xlsx'

print("Loading datasets...")
df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

cols = ['id', 'lat', 'long']
combined_df = pd.concat([df_train[cols], df_test[cols]], axis=0).drop_duplicates(subset=['id'])

print(f"Total unique properties to download: {len(combined_df)}")

# --- THE DOWNLOADER ---
def fetch_mapbox_image(lat, long, key):
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{long},{lat},17,0/224x224?access_token={key}"
    return requests.get(url)

print("Starting Mapbox download...")
success_count = 0
error_count = 0

# Loop through all properties
for index, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0]):
    str_id = str(row['id'])
    file_path = os.path.join(OUTPUT_DIR, f"{str_id}.jpg")
    
    if os.path.exists(file_path):
        continue

    try:
        response = fetch_mapbox_image(row['lat'], row['long'], API_KEY)

        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            success_count += 1
        else:
            print(f"Failed ID {str_id}: Status {response.status_code}")
            error_count += 1
            
    except Exception as e:
        print(f"Error on ID {str_id}: {e}")
        error_count += 1
     

print(f"Download process finished.")
print(f"Success: {success_count} | Errors: {error_count}")

# --- CHECK ---
num_files = len(os.listdir(OUTPUT_DIR))
print(f"Total files in directory: {num_files}")

# --- ZIP FOR EXPORT ---
if num_files > 100: 
    print("Zipping images... (This may take a minute)")
    shutil.make_archive('/kaggle/working/images_dataset', 'zip', OUTPUT_DIR)
    print("DONE! Download 'images_dataset.zip' from the Output tab.")
else:
    print("Something went wrong. Too few images downloaded.")
