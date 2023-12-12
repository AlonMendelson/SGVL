import os

cur_dir = os.getcwd()
path = os.path.join(cur_dir,"BLIP")
# Step 2: Read the existing content of the Python file (s.py)
file_paths = ["BLIP/models/blip_retrieval_vg.py","BLIP/train_retrieval_vg.py","BLIP/models/med.py","BLIP/models/vit.py","BLIP/vsr/evaluate_vsr.py","BLIP/Winoground/evaluate_winoground.py"]
for file_path in file_paths:
    with open(file_path, 'r') as file:
        content = file.read()

    # Step 3: Modify the content to insert the line at the beginning
    modified_content = f"import sys\nsys.path.insert(0, '{path}')\n{content}"

    # Step 4: Write the modified content back to the Python file
    with open(file_path, 'w') as file:
        file.write(modified_content)

