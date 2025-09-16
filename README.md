# Airplane Simulator

This project is a simulation environment for airlift logistics.

## Local Development Setup

These instructions will guide you through setting up the project for local development.

### Prerequisites

*   You must have a Conda installation (Miniconda or Anaconda).

### Installation Steps

1.  **Create the Conda Environment**

    This command will create a new Conda environment named `airlift` with all the required dependencies specified in the `environment.yml` file.

    ```bash
    conda env create -f airlift-starter-kit/environment.yml
    ```

2.  **Install the `airlift` Package in Editable Mode**

    This command installs the `airlift` project in "editable" mode. This is useful for development as any changes you make to the `airlift` source code will be immediately available in your environment.

    We use `conda run` to execute the command within the `airlift` environment without having to activate it first.

    ```bash
    conda run -n airlift pip install -e ./airlift
    ```

3.  **Activate the Conda Environment**

    To start working on the project, you need to activate the Conda environment in your terminal.

    ```bash
    conda activate airlift
    ```

    Your terminal prompt should now indicate that you are in the `airlift` environment. You are now ready to run the simulation and work on the code.
