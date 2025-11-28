Drew Schlesener and Denton Jarvis
GPGPU Final Project

This project is executed through a Jupyter Notebook in a Conda Environment on Palmetto

To run this project correctly:

1. Create a Jupyter Notebook Interactive Session on Palmetto with:
        * 4 CPU cores
        * 16 GB Memory
        * 1 A100 GPU
        Environment: Standard Jupyter Notebook
        Anaconda Version: anaconda3/2023.09-0
        Modules to be loaded: anaconda3 cuda

2. Dont worry about Conda environment, Run First cell of spectrogram_cuda.ipynb to set up everything needed for execution. 
    * After first cell has finished execution, click button top right of Jupityr Session named ex: Python (ipykernel), and change it to Python (spectrogram-cuda). This ensures your Notebook will execute within the environment. 
    
3. Then run Second cell to install correct CuPy runtime, ignore warning about multiple installations of Cuda

4. Then proceed through Jupityr Notebook and run each cell sequentially to see proper steps of Spectrogram generation. 