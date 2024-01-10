# Metabolic dysfunction-associated steatohepatitis (MASH) AI

## Installation
```bash
pip install -r requirements.txt
```

## Getting started
Check the notebook `Getting started.ipynb` or run on [Colab](https://colab.research.google.com/drive/1y8CcnN_y9dkk6OqZjRIsbEoEmuayYJFW?usp=sharing) to see the full usage. 

## Make predictions
Before making predictions for external validation, make sure of the following:
1. Data source is a .csv file
2. **Only** the following columns with the **exact** names are present:
   - ALANINE AMINOTRANSFERASE (ALT) (U/L)
   - GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)
   - PLATELET COUNT (1000 CELLS/UL)
   - AGE IN YEARS AT SCREENING
   - BODY MASS INDEX (KG/M**2)
3. Know which MASH AI model to use. Either MASH AI 35 trained with FAST >= 0.35 threshold or MASH AI 67 trained with FAST >= 0.67 threshold. 
4. The script `predict.py` will return the predicted label (1 or 0) and the probability of prediction (1 to 0), where 1 is at-risk MASH and 0 is *no* at-risk MASH

```bash
python predict.py {YOUR_DATA_SOURCE_HERE.csv} {MODEL_NAME_HERE}
```
Here's an example of the same data used for training. This is NOT a true external validation. This is for demostration only using the MASH AI 35 model.
```bash
python predict.py sample.csv xgboost_mashai_35
```

## Updates
**01/09/2024:** Will continue to test hyperparam optimization with geometric vs. harmonic means of sen, spec, ppv, and npv. 

**MASH AI 67**: The model underwent hyperparameter optimization with 1000 training iterations to maximize the geometric mean of sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV). The mean metrics achieved were as follows:
   - AUROC: 0.8945261437908496
   - Accuracy: 0.9854651162790697
   - Sensitivity: 0.75
   - Specificity: 0.9882352941176471
   - PPV: 0.42857142857142855
   - NPV: 0.9970326409495549

**MASH AI 35**: The model underwent hyperparameter optimization with 1000 training iterations to maximize the harmonic mean of sensitivity, specificity, PPV, and NPV. The mean metrics achieved were as follows:
   - AUROC: 0.9103909465020577
   - Accuracy: 0.8972868217054264
   - Sensitivity: 0.6833333333333333
   - Specificity: 0.9104938271604939
   - PPV: 0.3203125
   - NPV: 0.9789823008849557

## Contributing
Contributions are welcome. Please submit a pull request with your changes.

## License
This project is licensed under the terms of the MIT license. See the `LICENSE` file for details.