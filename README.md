# CIIBS - AI Cargo Inspection System

## Setup

### 1. Clone the repo
git clone https://github.com/Vinit3110/ciibs.git
cd ciibs

### 2. Create conda environment
conda create -n ciibs python=3.10 -y
conda activate ciibs

### 3. Install dependencies
pip install -r requirements.txt

### 4. Download model weights
[PASTE DRIVE LINK HERE]

Place it inside:
weights/best.pt

## Run
python scripts/app.py
