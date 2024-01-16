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
        print(f"Number of subjects dropped after NaN VCTE: {initial_unique_ids - data['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Apply condition mask
        condition_mask = (data['ELASTOGRAPHY EXAM STATUS'] == 1) & \
                         (data['COUNT:MEASURES ATTEMPTED WITH FINAL WAND'] >= 10) & \
                         (data['STIFFNESS E INTERQUARTILE RANGE (IQRE)'] < 30)
        data.loc[:, 'isGoodFibroScan'] = np.where(condition_mask, 1, 0)
        print("New column added: isGoodFibroscan for patients who met the following:")
        print("1. Elastography exam status = Completed, and")
        print("2. Exam with >= 10 measurements, and")
        print("3. IQR of stiffness < 30")
        print(f"Number of unique respondents meeting criteria: {data[data['isGoodFibroScan'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        return data

    @staticmethod
    def calculate_fast_score(data):
        """Calculates the FAST score."""
        print("FAST score is calculated using the formula: -1.65 + 1.07 * log(MEDIAN VCTE LSM) + 2.66e-8 * (MEDIAN CAP)^3 - 63.3 * (AST)^-1")
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
        print(f"New column added: {risk_column}")
        print(f"Number of unique respondents at risk: {data[data[risk_column] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")
        return data.drop(['FASTscore'], axis=1)


class DemographicsDataHandler:
    @staticmethod
    def process_demographics_data(df):
        """Processes demographic data with various criteria."""
        initial_unique_ids = df['RESPONDENT SEQUENCE NUMBER'].nunique()

        # Filter for age > 18 years
        df = df[df['AGE IN YEARS AT SCREENING'] >= 18]
        print(f"Number of subjects dropped who are <18 years old: {initial_unique_ids - df['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Creating new columns
        df['isNotViralHepatitis'] = ((df['HEPATITIS B SURFACE ANTIBODY'] != 1) | (df['HEPATITIS C ANTIBODY (CONFIRMED)'] != 1))
        print("New column added: isNotViralHepatitis for Hep B and Hep C")
        print(f"Number of unique respondents without viral hepatitis: {df[df['isNotViralHepatitis'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        df['Alcohol_g_per_day'] = df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] * 14
        print("New column added: Alcohol_g_per_day")
        print(f"Number of unique respondents with alcohol consumption data: {df[df['Alcohol_g_per_day'].notna()]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Determine high alcohol consumption
        conditions = {
            'isHighAlcoholConsumptionGT': ((df['GENDER'] == 1) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 2)) | ((df['GENDER'] == 2) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 1)),
            'isHighAlcoholConsumptionGTET': ((df['GENDER'] == 1) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] >= 2)) | ((df['GENDER'] == 2) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] >= 1))
        }
        for col, cond in conditions.items():
            df[col] = np.where(cond, 1, 0)
            print(f"New column added: {col} for women and men with an average of >1 and >2 drinks/day, respectively, over the past 12 months")
            print(f"Number of unique respondents with high alcohol consumption: {df[df[col] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        return df

class ClinicalDataTransformation:
    @staticmethod
    def transform_calculate_scores(data, scores=['HOMAIR','FIB4','NFS','APRI','BARD']):
        for score in scores:
            if score == 'FIB4':
                print("Calculating " + score)
                data = data.assign(FIB4 = lambda x: ( x['AGE IN YEARS AT SCREENING']*x['ASPARTATE AMINOTRANSFERASE (AST) (U/L)'] / x['PLATELET COUNT (1000 CELLS/UL)']*np.sqrt(x['ALANINE AMINOTRANSFERASE (ALT) (U/L)']) ) ) 
            elif score == 'HOMAIR':
                print("Calculating " + score)
                data = data.assign(HOMAIR = lambda x: ( x['INSULIN (μU/ML)'] * x['FASTING GLUCOSE (MG/DL)'] / 405) )
            elif score == 'NFS':
                print("Calculating " + score)
                data = data.assign(NFS = lambda x: ( x['AGE IN YEARS AT SCREENING']*0.037 + x['BODY MASS INDEX (KG/M**2)']*0.094 + x['DOCTOR TOLD YOU HAVE DIABETES']*1.13 + x['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']/x['ALANINE AMINOTRANSFERASE (ALT) (U/L)']*0.99 - x['PLATELET COUNT (1000 CELLS/UL)']*0.013 - x['ALBUMIN, REFRIGERATED SERUM (G/DL)']*0.66 ) )
            elif score == 'APRI':
                print("Calculating " + score)
                data = data.assign(APRI = lambda x: ( x['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']*40 / x['PLATELET COUNT (1000 CELLS/UL)'] )) 
            elif score == 'BARD':
                print("Calculating " + score)
                data = data.assign(BARD = lambda x: (1*(x['BODY MASS INDEX (KG/M**2)'] >= 28).astype(int) + 2*(x['ASPARTATE AMINOTRANSFERASE (AST) (U/L)']/x['ALANINE AMINOTRANSFERASE (ALT) (U/L)'] >= 0.8).astype(int) + 1*(x['DOCTOR TOLD YOU HAVE DIABETES'] == 1).astype(int)))

        return data

    @staticmethod
    def add_isDM_column(data):
        data['isDM'] = data[['GLYCOHEMOGLOBIN (%)', 'DOCTOR TOLD YOU HAVE DIABETES']].apply(lambda x: 1 if x['GLYCOHEMOGLOBIN (%)'] >= 6.5 and x['DOCTOR TOLD YOU HAVE DIABETES'] == 1 else 0, axis=1)
        print("New column added: isDM")
        print(f"Number of unique respondents with diabetes: {data[data['isDM'] == 1]['RESPONDENT SEQUENCE NUMBER'].nunique()}")
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
