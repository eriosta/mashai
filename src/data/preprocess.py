import numpy as np
import pandas as pd

class VCTEDataHandler:
    @staticmethod
    def filter_non_nan_values(data, columns):
        """Removes rows with NaN values in specified columns."""
        non_nan_mask = data[columns].notna().all(axis=1)
        return data[non_nan_mask]

    @staticmethod
    def get_acceptable_vcte(data):
        """Filters VCTE data based on specific criteria and adds a 'isGoodFibroScan' column."""
        initial_unique_ids = data['RESPONDENT SEQUENCE NUMBER'].nunique()

        # Filter out rows with NaN values in specific columns
        columns_to_check = ['MEDIAN STIFFNESS (E), KILOPASCALS (KPA)',
                            'MEDIAN CAP, DECIBELS PER METER (DB/M)']
        data = VCTEDataHandler.filter_non_nan_values(data, columns_to_check)
        print(f"* Subjects with NaN VCTE, N: {initial_unique_ids - data['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Apply condition mask
        condition_mask = (data['ELASTOGRAPHY EXAM STATUS'] == 1) & \
                         (data['COUNT:MEASURES ATTEMPTED WITH FINAL WAND'] >= 10) & \
                         (data['STIFFNESS E INTERQUARTILE RANGE (IQRE)'] < 30)
        data.loc[:, 'isGoodFibroScan'] = np.where(condition_mask, 1, 0)
        print(f"* Subjects without appropriate FibroScan measurements (<10 measurements and IQR LSM >=30%), N: {initial_unique_ids - data[data['isGoodFibroScan'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        return data

    @staticmethod
    def calculate_fast_score(data):
        """Calculates the FAST score."""
        print("* FAST scores calculated using the formula: FAST = -1.65 + 1.07 * log(MEDIAN STIFFNESS (E), KILOPASCALS (KPA)) + 2.66*(10**-8) * (MEDIAN CAP, DECIBELS PER METER (DB/M))^3 - 63.3 * (ASPARTATE AMINOTRANSFERASE (AST) (U/L))^-1")
        exponent = -1.65 + 1.07 * np.log(data['MEDIAN STIFFNESS (E), KILOPASCALS (KPA)']) + \
                   2.66*(10**-8) * data['MEDIAN CAP, DECIBELS PER METER (DB/M)']**3 - \
                   63.3 * data['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']**-1
        return np.exp(exponent) / (1 + np.exp(exponent))

    @staticmethod
    def add_at_risk_mash(data, cutoff):
        """Adds a column to indicate risk level based on the FAST score and a cutoff."""
        if cutoff not in [0.35, 0.67]:
            print("Please enter a valid cutoff value.")
            return data

        data['FASTscore'] = VCTEDataHandler.calculate_fast_score(data)
        risk_column = f'isAtRiskMASH{int(cutoff * 100)}'
        data[risk_column] = np.where(data['FASTscore'] >= cutoff, 1, 0)
        return data.drop(['FASTscore'], axis=1)


class DemographicsDataHandler:
    @staticmethod
    def process_demographics_data(df):
        """Processes demographic data with various criteria."""
        initial_unique_ids = df['RESPONDENT SEQUENCE NUMBER'].nunique()

        # Filter for age > 18 years
        df = df[df['AGE IN YEARS AT SCREENING'] >= 18]
        print(f"* Subjects <18 y/o, N: {initial_unique_ids - df['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Creating new columns
        df['isNotViralHepatitis'] = ((df['HEPATITIS B SURFACE ANTIBODY'] != 1) | (df['HEPATITIS C ANTIBODY (CONFIRMED)'] != 1))
        print(f"* Subjects with viral hepatitis, N: {initial_unique_ids - df[df['isNotViralHepatitis'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        df['Alcohol_g_per_day'] = df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] * 14

        # Determine high alcohol consumption
        conditions = {
            'isHighAlcoholConsumptionGT': ((df['GENDER'] == 1) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 2)) | ((df['GENDER'] == 2) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 1)),
        }
        for col, cond in conditions.items():
            df[col] = np.where(cond, 1, 0)
            print(f"* Women and men with an average of >1 and >2 drinks/day, respectively, over the past 12 months, N: {df[df[col] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        return df

class ClinicalDataTransformation:
    @staticmethod
    def transform_calculate_scores(data, scores=['HOMAIR','FIB4','NFS','APRI','BARD'], impute=False):
        if impute:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            columns_to_impute = ['AGE IN YEARS AT SCREENING', 'GENDER', 'GLYCOHEMOGLOBIN (%)', 'DOCTOR TOLD YOU HAVE DIABETES', 'BODY MASS INDEX (KG/M**2)', 'WAIST CIRCUMFERENCE (CM)', 'ASPARTATE AMINOTRANSFERASE (AST) (U/L)', 'ALANINE AMINOTRANSFERASE (ALT) (U/L)', 'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 'INSULIN (μU/ML)', 'PLATELET COUNT (1000 CELLS/UL)']
            imputed_data = data.copy()
            imputed_data[columns_to_impute] = imputer.fit_transform(imputed_data[columns_to_impute])

        for score in scores:
            if score == 'FIB4':
                print("...Calculating " + score + " using formula: (Age*AST) / (Platelet Count*sqrt(ALT))")
                data['FIB4'] = (imputed_data['AGE IN YEARS AT SCREENING']*imputed_data['ASPARTATE AMINOTRANSFERASE (AST) (U/L)'] / imputed_data['PLATELET COUNT (1000 CELLS/UL)']*np.sqrt(imputed_data['ALANINE AMINOTRANSFERASE (ALT) (U/L)']))
            elif score == 'HOMAIR':
                print("...Calculating " + score + " using formula: (Insulin*Fasting Glucose) / 405")
                data['HOMAIR'] = (imputed_data['INSULIN (μU/ML)'] * imputed_data['FASTING GLUCOSE (MG/DL)'] / 405)
            elif score == 'NFS':
                print("...Calculating " + score + " using formula: (Age*0.037 + BMI*0.094 + Diabetes*1.13 + AST/ALT*0.99 - Platelet Count*0.013 - Albumin*0.66)")
                data['NFS'] = (imputed_data['AGE IN YEARS AT SCREENING']*0.037 + imputed_data['BODY MASS INDEX (KG/M**2)']*0.094 + imputed_data['DOCTOR TOLD YOU HAVE DIABETES']*1.13 + imputed_data['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']/imputed_data['ALANINE AMINOTRANSFERASE (ALT) (U/L)']*0.99 - imputed_data['PLATELET COUNT (1000 CELLS/UL)']*0.013 - imputed_data['ALBUMIN, REFRIGERATED SERUM (G/DL)']*0.66)
            elif score == 'APRI':
                print("...Calculating " + score + " using formula: (AST*40) / Platelet Count")
                data['APRI'] = (imputed_data['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']*40 / imputed_data['PLATELET COUNT (1000 CELLS/UL)'])
            elif score == 'BARD':
                print("...Calculating " + score + " using formula: 1*(BMI >= 28) + 2*(AST/ALT >= 0.8) + 1*(Diabetes == 1)")
                data['BARD'] = (1*(imputed_data['BODY MASS INDEX (KG/M**2)'] >= 28).astype(int) + 2*(imputed_data['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']/imputed_data['ALANINE AMINOTRANSFERASE (ALT) (U/L)'] >= 0.8).astype(int) + 1*(imputed_data['DOCTOR TOLD YOU HAVE DIABETES'] == 1).astype(int))

        return data

    @staticmethod
    def add_isDM_column(data, impute=False):
        if impute:
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            columns_to_impute = ['GLYCOHEMOGLOBIN (%)', 'DOCTOR TOLD YOU HAVE DIABETES']
            imputed_data = data.copy()
            imputed_data[columns_to_impute] = imputer.fit_transform(imputed_data[columns_to_impute])

        data['isDM'] = imputed_data[['GLYCOHEMOGLOBIN (%)', 'DOCTOR TOLD YOU HAVE DIABETES']].apply(lambda x: 1 if x['GLYCOHEMOGLOBIN (%)'] >= 6.5 and x['DOCTOR TOLD YOU HAVE DIABETES'] == 1 else 0, axis=1)
        print(f"* Note: Subjects with diabetes mellitus, N: {data[data['isDM'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")
        return data

class ColumnsTextDataHandler:
    @staticmethod
    def parse_txt_file_to_dict(file_path):
        """Parses a text file into a dictionary."""
        with open(file_path, 'r') as file:
            lines = file.readlines()

        parsed_dict = {}
        current_key = None
        for line in lines:
            if line.strip() and line[0].isdigit() and "." in line:
                current_key = line.split(".")[1].strip()
                parsed_dict[current_key] = {}
            elif "-" in line and current_key:
                parts = line.split(" - ", 1)
                if len(parts) == 2:
                    key, value = parts
                    parsed_dict[current_key][key.strip()] = value.strip()

        return parsed_dict
