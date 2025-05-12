import subprocess

style_id = 0  # Set the specific style_id you want to use

command = [
        'python', 'inference/generate_face.py',  
        '--style', 'makeup',  
        '--name', 'makeup',    
        '--style_id', str(style_id),
        '--weight', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',
        '--output_path', './output/makeup/',  
        "--content",'./data/makeup/images/test/003767.png', 
        "--makeup_name",'refined_makeup_code.npy', 
        "--align_face"
        ]
print("Running style_transfer with style_id:", style_id) 
subprocess.run(command)
