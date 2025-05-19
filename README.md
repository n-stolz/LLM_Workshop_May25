## LLM Workshop

Welcome to the workshop on how to use Large Language Models (LLM) in research.

Here is an overview of the files in the repository:

- **`llm_workshop_python.ipynb`**: This is the notebook you will be working with.
- **`reports/`**: Directory where the documents we are analyzing are saved. You can open them, but **please don't rename them**.
- **`app3.py`**: This is the code for the web interface — no need to open or change it.


To get started:\n
1. Clone this repository on your computer: 

 
```bash
git clone https://github.com/n-stolz/LLM_Workshop_May25.git
cd LLM_Workshop_May25
```


2. Create a virtual environment

    For windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```


    For Mac:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Then install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. You should have gotten the file "key_file.env" in a separate email from the workshop administrator. Copy it to the git folder. Do not change the name of this file - it is in the .gitignore file, so it will not be commited to github.
