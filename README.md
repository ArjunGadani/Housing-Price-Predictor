# Housing-Price-Predictor
- Housing Price Predictor is machine learning model to predict price of house depends on various parameters ( Such as bedrooms, floors, garage, garageSpace, neighbourhood, etc. ) which are provided to model. I Have trained this model on ```Boston House Price``` dataset.
- By applying various regression models, I have found best accuracy using Random Forest Regressor model and i have trained model using cross-validation to avoid over-fitting issue.

###### Required Libraries are defined in ```requirement.txt``` File.

``` pip install -r requirement.py ```

Run this command on terminal to install required Libraries to run Model.

### To Run:
- Open Terminal in root location of project and dataset is provided in ```\data``` Directory.

```python Model.py```

- Final Output Score are:

![Scores](https://user-images.githubusercontent.com/105216607/180648560-61ad6378-d726-4279-b1c9-f13230dc9295.JPG)

- It will save model using Joblib Module, So you dont have to train data every time.
