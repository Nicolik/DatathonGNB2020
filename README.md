# GNB 2020 Datathon

## Team 06 Final Submission
Team Members:
Nicola Altini (Team Leader), Chiara Roversi (Co-Leader), 
Franca Scocozza, Fazeelat Mazhar, Walter Baccinelli, Paola De Stefano, 
Bianca Barzaghini, Claudio Pighini, Lorenzo Coppadoro, Emir Mobedi, Tatiana Mencarini.


## General description
Python code processing data from GNB 2020 Datathon.

The GNB 2020 Datathon focused on the development of mortality prediction 
models for intensive care unit (ICU) patients, starting from routinely collected data in the first 48 hours of ICU stay.
An already pre-processed data-set is given to participants, containing aggregating features extracted from the original time 
series, with the aim to predict in-hospital mortality.

10 teams were involved in the competitions. Our model had the highest evaluation metrics, that were 
the minimum of Sensitivity and Precision, the Area Under Precision Recall Curve and the Area Under Receiving Operating Curve.


## Usage
run ``train.py`` for training our ensemble(5) model.
run ``test.py`` to assess performance on the test set (released by the organizers after the end of the challenge).


### Main steps of the code
At first, some feature engineering has been performed on the dataset:
- features with more than 80% of NaNs have been removed;
- missing values have been imputed with a KNN algorithm;
- features have been scaled to a fixed interval.

The developed model consists in a weighted ensemble of 5 different simpler models, that are Logistic Regression, Support Vector 
Machine with RBF Kernel, Random Forests, Ada Boost and Multi Layer Perceptron. The classification threshold was computed by
maximizing the most important metric of the competition, i.e. the minimum of Sensitivity and Precision.
A 10-fold cross-validation was performed on training set, yielding a minimum of Sensitivity and Precision of 0.502 +/- 0.036,
an Area Under Precision Recall Curve of 0.534 +/- 0.034 and an Area Under Receiving Operating Curve of 0.859 +/- 0.011.
On the test set, the model obtained, respectively, 0.565, 0.550 and 0.866 on the same evaluation metrics.

## Citation
Use this bibtex to cite this repository:
```
@misc{nicolik_gnb_datathon_2020,
  title={Ensemble(5) for Mortality Risk Prediction in ICU Domain},
  author={Altini, Nicola and Roversi, Chiara and Scocozza, Franca and Mazhar, Fazeelat and Baccinelli, Walter and De Stefano, Paola and
          Barzaghini, Bianca and Pighini, Claudio and Coppadoro, Lorenzo and Mobedi, Emir and Mencarini, Tatiana},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Nicolik/DatathonGNB2020}},
}
```
