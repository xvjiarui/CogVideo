"""
python tools/openvid/unzip_openvid.py /home/jiaruix/new_home/jiarui-data/openvid/OpenVid-1M/ /home/jiaruix/new_home/jiarui-data/openvid/videos/
"""
import os
import argparse
import zipfile
import shutil
import tqdm

def concatenate_files(parts, output_file):
    if os.path.exists(output_file):
        print(f"Skipping concatenation, {output_file} already exists.")
        return
    with open(output_file, 'wb') as concat_file:
        for part in parts:
            with open(part, 'rb') as part_file:
                concat_file.write(part_file.read())

def extract_zip_file(zip_file_path, output_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        
        if os.path.exists(output_folder):
            zip_file_count = len(zip_ref.namelist())
            extracted_file_count = sum([len(files) for r, d, files in os.walk(output_folder)])
            if extracted_file_count == zip_file_count:
                print(f"Skipping extraction, {output_folder} already contains all files.")
                return
            else:
                print(f"Folder {output_folder} already exists, but does not contain all files, extracted {extracted_file_count} out of {zip_file_count} files.")
        
        FILES_PER_FOLDER = 1000
        namelist = zip_ref.namelist()
        total_subfolders = len(namelist) // FILES_PER_FOLDER + 1
        # Show progress bar
        with tqdm.tqdm(total=len(namelist), desc="Extracting files", position=1) as pbar:
            for idx, file in enumerate(namelist):
                # Skip directories
                if file.endswith('/'):
                    continue
                
                # Extract each file without recreating the directory structure
                source = zip_ref.open(file)
                subfoler_name = f'part_{idx // FILES_PER_FOLDER:03d}_of_{total_subfolders:03d}'
                target_path = os.path.join(output_folder, subfoler_name, os.path.basename(file))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                pbar.update(1)

def main(folder_path, output_folder):
    # List all files in the folder
    files = sorted(os.listdir(folder_path))

    # Group files that need to be concatenated
    concat_groups = {}
    for file in files:
        if 'part' in file and not file.endswith('.zip'):
            base_name = '_'.join(file.split('_')[:-1])
            if base_name not in concat_groups:
                concat_groups[base_name] = []
            concat_groups[base_name].append(os.path.join(folder_path, file))
        elif file.endswith('.zip'):
            base_name = file.split('.')[0]
            concat_groups[base_name] = [os.path.join(folder_path, file)]

    # Process each group
    progress_bar = tqdm.tqdm(range(len(concat_groups)), desc="Processing files")
    for base_name, parts in concat_groups.items():
        progress_bar.set_description(f"Processing {base_name}")
        if len(parts) > 1:
            progress_bar.set_description(f"Processing {base_name} (concatenating)")
            # Concatenate parts
            concat_file_path = os.path.join(folder_path, f"{base_name}.zip")
            concatenate_files(parts, concat_file_path)
            progress_bar.set_description(f"unzipping {base_name}")
            # Unzip the concatenated file
            extract_zip_file(concat_file_path, os.path.join(output_folder, base_name))
        else:
            progress_bar.set_description(f"unzipping {base_name}")
            # Unzip single part file
            extract_zip_file(parts[0], os.path.join(output_folder, base_name))
        progress_bar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip and concatenate OpenVid zip files.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the zip files.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where files will be unzipped.")
    args = parser.parse_args()
    
    main(args.folder_path, args.output_folder)