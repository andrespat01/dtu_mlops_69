# mlops_project

MLOPS project 

## Project structure

The directory structure of the project looks like this:
```txt
├── .dvc
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── app/
│   └── app/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── configs/                  # Configuration files
│   ├── config_cpu.yaml
│   ├── config.yaml
│   └── sweep.yaml
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
│   └── model.pth
│
├── prediction_api/           # Source code
│   ├── api/
│   │   ├── main.py           # The function uploaded to cloud run
│   │   └── requirements.txt  # The requirements to run the function
│   ├── prediction_database.csv # Database of the calls to API: timestamp, location, text and prediction
│   └── read_me.txt           # Explains how to use the api
│
├── reports/                  # Reports
│   └── figures/
│   │    └── Confusion_matrix.png
│   ├── README.md
│   └── report.md
├── src/                      # Source code
│   └── mlops_project/
│       ├── __init__.py
│       ├── api.py            # API call
│       ├── data_drift.py     # Data drifting
│       ├── data.py           # Receive data from cloud and preprocess
│       ├── frontend.py       # Frontend of the program
│       ├── models.py         # Train model using Pytorch Lightning
│       └── visualize.py      # Generate confusion matrix
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .coverage
└── wandb/                    # Wandb folder
│   └── latest-run/
│       ├── files/...
│       ├── logs/...
│       └── temp/...
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── coverage_report.txt       # Coverage report
├── environment.yml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── report.html
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
├── setup.py                  # 
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# Code coverage:
pytest --cov=tests/ --cov-report=term-missing > coverage_report.txt