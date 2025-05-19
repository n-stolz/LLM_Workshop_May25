## LLM Workshop

Welcome to the workshop on how to use Large Language Models (LLM) in research.

Here is an overview of files in the repository: \n
    - llm_workshop_python.ipynb : this is the notebook you will be working with
    - reports: directory where documents we are analysing are saved. You can open them, but please don't rename them
    - app3.py: This is the code for the web interface - no need to open or change it


To get started:\n
1. Clone this repository on your computer: 

   git clone https://github.com/n-stolz/LLM_Workshop_May25.git

2. Create a virtual environment

    For windows:
    python -m venv venv
    venv\Scripts\activate

    For Mac:
    python3 -m venv venv
    source venv/bin/activate

3. Then install dependencies:
    pip install -r requirements.txt

4. You should have gotten the file "key_file.env" in a separate email from the workshop administrator. Copy it to the git folder. Do not change the name of this file - it is in the .gitignore file, so it will not be commited to github.
