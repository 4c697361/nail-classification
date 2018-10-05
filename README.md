Nail classification
==============================

A deep learning demo project for classifying manufactured nails.  

The data set used for this demo contains 100 images of 'good' nails and 100 images of 'bad' nails, where the property 'good' and 'bad' refer to the nail being either intact or somehow bent, respectively.  

As this data set is rather small it is highly recommended to use a pre-trained model as the building block of the classifier.  In this demo, the pre-trained [vgg16](https://keras.io/applications/#vgg16) model implemented in keras is used along with a customized top to accomplish this binary classification task.  

Additionally, a simple CNN is implemented to establish an easy baseline.  The simple CNN reached a validation accuracy of 
*0.625 %*
after training for 65 epochs 
whereas the pre-trained vgg16-architecture achived 
*0.958 %*
for the validation accuracy after training for 10 epochs.  


To increase the performance of the models a cropping algorithm was applied upon the images.  Thereby only the region which shows the target is considered during training and for the prediction.  


# 1. Easy start - out of the box nail classification: 
1. Clone the repository

    ```console
    git clone git@github.com:L2Data/nail-classification.git
    cd nail-classification
    ```
    
2. Place the pre-trained model in the folder models/.
   When you have not any pre-trained model, section **3.** explains how to train you model.
   
3. **Using docker**<br/>
    Build the docker image (if docker is not installed yet: see the [docker documentation](https://docs.docker.com/) for instructions)<br/>
        
    ```console
    docker build -t nail-classifier .
    docker run -p 127.0.0.1:5000:5000 nail-classifier    
    ```
        
    This starts the server API with the pre-trained model in the terminal.  <br/>
    For classifying an image, open a new terminal and type<br/> 

    ```console
    curl -X POST -F image=@<path-to-your-nail-image.jpeg> 'http://localhost:5000/predict' 
    ```

    The classifier is now predicting the class of the image.  <br/>
    Simultaneously, it will give you information about the probability that the image belongs to 

        a. class 0:  bad nails, p_bad = 1-p_good        
        b. class 1:  good nails, p_good

    The output format is JSON.
    
**Alternative to docker:  local usage**  <br/>
You can run the server without the docker image.  Execute<br/>

```console
make server   
```
in the root directory of the project starts the server.  <br/>
Again, open a new terminal and run 

```console
curl -X POST -F image=@<path-to-your-nail-image.jpeg> 'http://localhost:5000/predict' 
```

to classify your nail image.  
    
        
# 2. More information:
------------
The local installation of this project comes with several options.  
Execute 

```console
make
```

in the root directory of the project to see what is available: 

```console
clean               Delete all compiled Python files 
create_environment  Set up python interpreter environment 
data                Make Dataset 
model_predict       Predict from trained model 
model_train         Train a model 
requirements        Install Python Dependencies 
server              Run API 
test_environment    Test python environment is setup correctly 
```

**REMARK:** <br/>
Using ```console make <command> ``` always executes the corresponding python script with default settings.  


# 3. Train a model from scratch:
----------
For this task first copy the images into the data folder.  The tree structure should look like 

    └── data
        └── raw
            └── nailgun
                ├── good
                └── bad   

and then execute
```console
make data
```
which creates 

    └── data
        └── raw
        │   └── nailgun
        │       ├── good
        │       └── bad
        └── processed
            ├── train
            |   ├── good
            |   └── bad
            ├── validate
            |   ├── good
            |   └── bad
            └── test
                ├── good
                └── bad

The subfolders in processed contain cropped images according to a set validation and test split.  
Alternatively, the data sets can be created by running 
```console
python src/data/make_dataset.py [OPTIONS]
[OPTIONS]:  --split (default 0.12: valdiation and test split)
            --seed  (default 42: random seed)
            --clean (default 1 (True): clean processed/<subdirs>)
            --crop  (default 1 (True): apply cropping)
```

After creating the training data sets the model can be trained.  Therefore execute 
```console
make model_train
```

Alternatively, the model can be trained by running 
```console
python src/models/train_model.py [OPTIONS]
[OPTIONS]:  --modelname (default 'vgg16: select the model, alternative: 'cnn')
            --ep
            --lr
            --augment
            --bs
            --width
            --heigth
```
where the default settings of all the options but the modelname are found in ```src/utils/utils.py```.  
The models are defined in ```src/models/model.py``` and you can add as many different architectures as you like.  

The training can be monitored with tensorboard by running 
```console
tensorboard --logdir=<path-to-project>/nail-classification/logs/
```
from the root directory of the project and executing 
```console
http://localhost:6006
```
in your browser.  

To run a prediction from the (pre-)trained model finally run 
```console
make model_predict
```
or, alternativly, 
```console
python src/models/train_predict.py [OPTIONS]
[OPTIONS]:  --modelname (default 'vgg16: select the model, alternative: 'cnn')
```
which yields ```probability(good), label, name``` as an output.  

# 4. Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# nail-classification
