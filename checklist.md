### Week 1

- [X] Create a git repository
- [X] Make sure that all team members have write access to the github repository
- [X] Create a dedicated environment for you project to keep track of your packages (using conda)
- [X] Create the initial file structure using cookiecutter
- [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
- [x] Add a model file and a training script and get that running
- [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
- [x] Remember to comply with good coding practices (`pep8`) while doing the project
- [x] Do a bit of code typing and remember to document essential parts of your code
- [] Setup version control for your data or part of your data
- [] Construct one or multiple docker files for your code
- [] Build the docker files locally and make sure they work as intended -> (training works assuming data has been downloaded and processed)
- [] Write one or multiple configurations files for your experiments
- [] Used Hydra to load the configurations and manage your hyperparameters
- [] When you have something that works somewhat, remember at some point to to some profiling and see if you can optimize your code
- [] Use wandb to log training progress and other important metrics/artifacts in your code -> (should be checked)
- [] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code


### Week 2
- [x] Write unit tests related to the data part of your code (M16)
- [x] Write unit tests related to model construction and or model training (M16)
- [] Calculate the code coverage (M16)
- [] Get some continuous integration running on the GitHub repository (M17)
- [] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
- [] Add a linting step to your continuous integration (M17)
- [] Add pre-commit hooks to your version control setup (M18)
- [] Add a continues workflow that triggers when data changes (M19)
- [] Add a continues workflow that triggers when changes to the model registry is made (M19)
- [] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
- [] Create a trigger workflow for automatically building your docker images (M21)
- [] Get your model training in GCP using either the Engine or Vertex AI (M21)
- [] Create a FastAPI application that can do inference using your model (M22)
- [] Deploy your model in GCP using either Functions or Run as the backend (M23)
- [] Write API tests for your application and setup continues integration for these (M24)
- [] Load test your application (M24)
- [] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
- [] Create a frontend for your API (M26)

### Week 3
- [] Check how robust your model is towards data drifting (M27)
- [] Deploy to the cloud a drift detection API (M27)
- [] Instrument your API with a couple of system metrics (M28)
- [] Setup cloud monitoring of your instrumented application (M28)
- [] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
- [] If applicable, optimize the performance of your data loading using distributed data loading (M29)
- [] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
- [] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)


### Extra
- [] Write some documentation for your application (M32)
- [] Publish the documentation to GitHub Pages (M32)
- [] Revisit your initial project description. Did the project turn out as you wanted?
- [] Create an architectural diagram over your MLOps pipeline
- [] Make sure all group members have an understanding about all parts of the project
- [] Uploaded all your code to GitHub