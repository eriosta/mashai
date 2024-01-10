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
        print(f"Unique RESPONDENT SEQUENCE NUMBER dropped after NaN filter: {initial_unique_ids - data['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        # Apply condition mask
        condition_mask = (data['ELASTOGRAPHY EXAM STATUS'] == 1) & \
                         (data['COUNT:MEASURES ATTEMPTED WITH FINAL WAND'] >= 10) & \
                         (data['STIFFNESS E INTERQUARTILE RANGE (IQRE)'] < 30)
        data.loc[:, 'isGoodFibroScan'] = np.where(condition_mask, 1, 0)

        return data

    @staticmethod
    def calculate_fast_score(data):
        """Calculates the FAST score."""
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
        print(f"Unique RESPONDENT SEQUENCE NUMBER dropped after age filter: {initial_unique_ids - df['RESPONDENT SEQUENCE NUMBER'].nunique()}")

        df['isNotViralHepatitis'] = ((df['HEPATITIS B SURFACE ANTIBODY'] != 1) | (df['HEPATITIS C ANTIBODY (CONFIRMED)'] != 1))
        df['Alcohol_g_per_day'] = df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] * 14

        # Determine high alcohol consumption
        conditions = {
            'isHighAlcoholConsumptionGT': ((df['GENDER'] == 1) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 2)) | ((df['GENDER'] == 2) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] > 1)),
            'isHighAlcoholConsumptionGTET': ((df['GENDER'] == 1) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] >= 2)) | ((df['GENDER'] == 2) & (df['AVG # ALCOHOLIC DRINKS/DAY - PAST 12 MOS'] >= 1))
        }
        for col, cond in conditions.items():
            df[col] = np.where(cond, 1, 0)

        return df


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
