# Airplane Simulator

This project is a simulation environment for airlift logistics.

## Local Development Setup

These instructions will guide you through setting up the simulation environment in an Linux environment.

### Prerequisites

*   You must have a Conda installation (Miniconda or Anaconda).

### Installation Steps

1.  **Create the Conda Environment**

    This command will create a new Conda environment named `airlift` with all the required dependencies specified in the `environment.yml` file.

    ```bash
    conda env create -f airlift-starter-kit/environment.yml
    ```

2.  **Install the `airlift` Package in Editable Mode**

    Activate the conda environment. 
    '''bash
    conda activate airlift-solution"
    '''

    Then run this installation to download the airlift software package. 

    ```bash
    pip install -e ./airlift
    ```

3.  **Activate the Conda Environment**

    To start working on the project, you need to activate the Conda environment in your terminal.

    ```bash
    conda activate airlift-solution
    ```

    Your terminal prompt should now indicate that you are in the `airlift` environment. You are now ready to run the simulation and work on the code.
