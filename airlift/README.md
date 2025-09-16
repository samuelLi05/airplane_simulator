# ✈️ Airlift Challenge Simulator

This simulation environment is part of the Airlift Challenge, a competition in which participants must design agents that can plan and execute an airlift operation.
Quick decision-making is needed to rapidly adjust plans in the face of disruptions along the delivery routes.
The decision-maker will also need to incorporate new cargo delivery requests that appear during the episode.
The primary objective is to meet the specified deadlines, with a secondary goal of minimizing cost.
Solutions can incorporate machine learning, optimization, path planning heuristics, or any other technique.

## Getting started
* To write your own solution: download the [Starter Kit](https://github.com/airlift-challenge/airlift-starter-kit) and follow the instructions in the README
* For more information about the competition and simulator: [Documentation](https://airlift-challenge.github.io/)
* For submissions and to participate in the discussion board: see the competition platform on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/0000)

## Installing with pip
The `airlift-challenge` simulator package is designed to work with Python version 3.9.
You can install the latest version of the package directly from this repo by running the following command:
```console
$ pip install git+https://github.com/airlift-challenge/airlift
```

For debugging purposes, you can also install the package from source:
```console
$ git clone https://github.com/airlift-challenge/airlift
$ cd airlift
$ pip install -e .
```

## Demo
Run the demo to test that the installation works and view a simple scenario:
```console
$ airlift-demo
```

## Unit tests
To run the full test suite execute the following command from within the airlift source code folder:
```console
pytest tests
```

# Credits
The development team consists of:
* ccafeccafe
* Adis Delanovic
* Jill Platts
* Alexa Loy
* Andre Beckus

This competition and code-base was heavily influenced by the [Flatland challenge](https://gitlab.aicrowd.com/flatland/flatland/), and incorporates some structure and code fragments from this competition.
We are grateful to the [companies and people](https://gitlab.aicrowd.com/flatland/flatland/#-credits) that created this great open source environment. 

Distribution Statement A: Approved for Public Release; Distribution Unlimited: Case Number: AFRL-2023-5590, CLEARED on 03 Nov 2023