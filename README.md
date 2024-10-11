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

## Contributing
Contributions are welcome. Please submit a pull request with your changes.

## License
This project is licensed under the terms of the MIT license. See the `LICENSE` file for details.