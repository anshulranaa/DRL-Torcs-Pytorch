import subprocess
import os

script_paths = ['PPO/ppo_run.py', 'DQN/dqn.py', 'DDPG-PPO/ddpg_ppo.py']

for script_path in script_paths:
    # Extract the directory path
    directory = os.path.dirname(script_path)
    script_file = os.path.basename(script_path)
    
    # Change to the directory
    os.chdir(directory)
    
    print(f"running {script_path}..")
    # Run the script
    result = subprocess.run(['python', script_file], capture_output=True, text=True)
    print("run finished.")
    # Change back to the original directory if needed
    os.chdir('..')  # Assuming you want to go back to the root directory of the project

