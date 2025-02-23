# Med-MNIST Project

## Running Locally

1. **Clone the Repository**  
   ```bash
   git clone <REPO_URL>
   ```
2. **Install Conda (if not installed)**  
   [Miniconda Download Link](https://docs.conda.io/en/latest/miniconda.html)
3. **Create the Environment**  
   ```bash
   conda env create -f environment.yml
   ```
4. **Open the Notebook**  
   - Use VSCode or any IDE that supports Jupyter notebooks.
   - Select the newly installed `med-mnist` environment as the kernel.
   - Run the notebook cells.

## Running on a Remote Docker (RunPod, Vast.ai, etc.)

1. **Public/Private SSH Key Setup**  
   - Generate an SSH key on your local machine (if you don’t have one).
   - Add the **public key** to your RunPod/Vast.ai account settings.

2. **Create a New Instance**  
   - Pick a normal Linux image (for example, `ubuntu:noble-20250127`).
   - Specify the following on-start script in the instance settings:

   ```bash
   #!/usr/bin/env bash

   # Usage:
   # 1. Copy this script to the On-start Script section
   # 2. Start the VM and wait for the setup to complete

   set -eux

   apt-get update

   ##
   # 1. Install Miniconda
   ##
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh

   # Activate Miniconda for this session
   source ~/miniconda3/bin/activate

   # Initialize Conda for all shell types
   ~/miniconda3/bin/conda init --all

   # Ensure Conda is available in future shell sessions
   # echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   # echo 'source ~/miniconda3/bin/activate' >> ~/.bashrc

   ##
   # Install Unison for File Sync
   ##
   apt-get update
   DEBIAN_FRONTEND=noninteractive apt-get install -y unison

   ##
   # Verify Installations
   ##
   echo "Miniconda Installed:"
   conda --version

   echo "Unison Installed:"
   unison -version

   echo "SSH Status:"
   systemctl status ssh | grep Active

   echo "Setup Complete! Miniconda and Unison are ready."
   ```

3. **Syncing Your Code with Unison**  
   From your local machine, use a Unison profile or a single command:

   **Profile Example** (`unison_medmnist.prf`):
   ```
    # Define local and remote sync locations
    root = <PATH>\med-mnist
    root = ssh://<USER>@<IP>:<PORT>//root/med-mnist

    # Sync the entire folder except ignored files
    ignore = Path datasets
    ignore = Path src/__pycache__
    ignore = .git

    # Ignore specific files
    ignore = Name README.md
    ignore = Name unison.prf
    ignore = Name .gitignore
    ignore = Name *.log
   ```

   Copy `unison_medmnist.prf` to your local Unison directory (for example, `C:\Users\<User>\.unison\`) and run:
   ```bash
   unison unison_medmnist.prf -auto -batch -repeat 2
   ```

   **One-Liner Command** (avoids using a `.prf` file):
   ```powershell
    unison "<PATH>\med-mnist" "ssh://<USER>@<IP>:<PORT>//root/med-mnist" -auto -batch -repeat 2 -ignore "Path datasets" -ignore "Path src/__pycache__" -ignore "Path .git" -ignore "Name README.md" -ignore "Name unison.prf" -ignore "Name .gitignore" -ignore "Name *.log"

   ```

4. **SSH and Port Forwarding**
   ```bash
   ssh -p <PORT> <USER>@<IP> -L 8080:localhost:8080
   ```
   The `-L 8080:localhost:8080` flag forwards port 8080 on your local machine to port 8080 on the remote machine. This lets you connect to services (like Jupyter) running remotely on port 8080 by visiting `localhost:8080` in your local browser.

5. **Create and Activate Conda Environment**
   ```bash
   cd /root/med-mnist
   conda env create -f environment.yml
   conda activate med-mnist
   ```

6. **Install Jupyter (if missing)**
   ```bash
   conda install jupyter
   ```

7. **Run Jupyter Server on the Remote**
   ```bash
   jupyter notebook --no-browser --port 8080 --allow-root
   ```
   Copy the localhost URL with the token from the terminal output.  
   Example: `http://127.0.0.1:8080/?token=<SOME_TOKEN>`

8. **Checking Running Jupyter Notebooks and Tokens**
   If you need to retrieve the Jupyter notebook token, then from another SSH session, run:
   ```bash
   jupyter notebook list
   ```
   This will show active Jupyter sessions along with their URLs and tokens.

9. **Connect from VS Code or Another IDE**
   - Open your notebook on your local machine.
   - Select “Remote Jupyter Server” and provide `http://127.0.0.1:8080/?token=<SOME_TOKEN>` as the server URL.
   - Run the notebook cells as if it were local.