import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact, shapiro, levene, t
from sklearn.impute import KNNImputer

class StatisticalAnalysis:
    def __init__(self, data, response_var):
        self.data = data
        self.response_var = response_var

    def summarize_stats_continuous(self, data=None, response_var=None):
        """
        Performs summary statistics between response_var==1 and response_var==0 and generates a table with the following:
        - variable name
        - mean difference
        - test type (t-test, Mann-Whitney U test, etc.)
        - statistic
        - p-value
        - confidence intervals
        - mean 1
        - mean 2

        :param data: pandas DataFrame
        """

        def bootstrap_mean_difference(group1, group2, n_iterations=1000, alpha=0.05):
            """
            Computes the bootstrap estimate of the difference in means and its confidence interval.
            
            :param group1: pandas Series or array-like
            :param group2: pandas Series or array-like
            :param n_iterations: int, number of bootstrap iterations
            :param alpha: float, significance level
            :return: tuple, (mean difference, (lower CI, upper CI))
            """
            np.random.seed(42)  # Set the seed for reproducibility
            data1 = group1.values
            data2 = group2.values
            mean_diffs = []

            for _ in range(n_iterations):
                sample1 = np.random.choice(data1, len(data1), replace=True)
                sample2 = np.random.choice(data2, len(data2), replace=True)
                mean_diffs.append(sample2.mean() - sample1.mean())

            mean_diffs = np.array(mean_diffs)
            mean_diff = mean_diffs.mean()
            ci_lower = np.percentile(mean_diffs, alpha / 2 * 100)
            ci_upper = np.percentile(mean_diffs, (1 - alpha / 2) * 100)

            return mean_diff, (ci_lower, ci_upper)

        if data is None:
            data = self.data
        if response_var is None:
            response_var = self.response_var

        variables = data.columns.drop(response_var)
        summary_rows = []  # Collect rows in a list

        for var in variables:
            # Split the data into two groups based on response variable
            group1 = data.loc[data[response_var] == 0, var]
            group2 = data.loc[data[response_var] == 1, var]
            # Compute the mean difference
            mean_diff = group2.mean() - group1.mean()
            # Check for normality and equal variances
            equal_var = False
            normality = False
            stat, pval = 0, 0
            ci = ''

            # Normality test
            if shapiro(group1)[1] > 0.05 and shapiro(group2)[1] > 0.05:
                normality = True

            # Equal variance test
            if levene(group1, group2)[1] > 0.05:
                equal_var = True

            # Perform appropriate test based on normality and equal variance
            test_type = ''
            if not equal_var and not normality:
                stat, pval = ttest_ind(group1, group2, equal_var=False, trim=0.2)
                test_type = "Yuen's trimmed t-test"

            elif equal_var and not normality:
                mean_diff, (ci_lower, ci_upper) = bootstrap_mean_difference(group1, group2)
                test_type = 'Bootstrap Mean Difference'
            elif not equal_var and normality:
                stat, pval = ttest_ind(group1, group2, equal_var=False)
                test_type = 'Welch\'s t-test'
            else:
                stat, pval = ttest_ind(group1, group2, equal_var=True)
                test_type = 'Independent t-test'

            # Compute confidence interval
            if test_type == 'Bootstrap Mean Difference':
                ci = f'[{ci_lower:.2f}, {ci_upper:.2f}]'
            else:
                diff_std = (group1.std() ** 2 / group1.size + group2.std() ** 2 / group2.size) ** 0.5
                se = diff_std * t.ppf(1 - 0.05 / 2, group1.size + group2.size - 2)
                lower_ci = mean_diff - se
                upper_ci = mean_diff + se
                ci = f'[{lower_ci:.2f}, {upper_ci:.2f}]'   

            # Create a row dictionary and add it to the list
            row = {
                'Variable': var, 
                'Mean difference': mean_diff, 
                'Test type': test_type, 
                'Statistic': stat, 
                'P-value': pval, 
                'Confidence interval': ci, 
                'Mean, No High-Risk MASLD': group1.mean(), 
                'Mean, High-Risk MASLD': group2.mean()
            }
            summary_rows.append(row)

        # Create DataFrame from the list of rows
        summary_table = pd.DataFrame(summary_rows)

        return summary_table

    def summarize_stats_categorical(self, data=None, response_var=None):
        if data is None:
            data = self.data
        if response_var is None:
            response_var = self.response_var

        variables = data.columns.drop(response_var)
        summary_rows = []

        for var in variables:
            # Create a contingency table
            table = pd.crosstab(data[var], data[response_var])
            total = table.sum().sum()

            statistic = None

            # Perform appropriate test
            if table.shape == (2, 2):
                oddsratio, pval = fisher_exact(table)
                test_type = 'Fisher\'s exact test'
                statistic = oddsratio
            else:
                chi2, pval, dof, expected = chi2_contingency(table)
                test_type = 'Chi-square test'
                statistic = chi2

            # Format the table output
            formatted_table = table.apply(lambda x: x.astype(str) + ' (' + (100 * x / total).round(2).astype(str) + '%)')
            
            # Format p-value
            formatted_pval = f'{pval:.3g}'

            row = {
                'Variable': var, 
                'Test type': test_type,
                'Statistic': statistic,
                'P-value': formatted_pval
            }

            summary_rows.append(row)

        # Create DataFrame from the list of rows
        summary_table = pd.DataFrame(summary_rows)
        return summary_table


df = pd.read_csv('data/processed/NhanesPrepandemicAllWithAST.csv').drop('Unnamed: 0', axis=1)

# Assuming these are the columns you're interested in
cols = ['AGE IN YEARS AT SCREENING', 'GENDER', 'RACE/HISPANIC ORIGIN W/ NH ASIAN', 'isDM', 'BODY MASS INDEX (KG/M**2)', 'WAIST CIRCUMFERENCE (CM)', 'ASPARTATE AMINOTRANSFERASE (AST) (U/L)', 'ALANINE AMINOTRANSFERASE (ALT) (U/L)', 'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 'ALBUMIN, REFRIGERATED SERUM (G/DL)', 'PLATELET COUNT (1000 CELLS/UL)', 'GLYCOHEMOGLOBIN (%)', 'FASTING GLUCOSE (MG/DL)', 'INSULIN (μU/ML)', 'DIRECT HDL-CHOLESTEROL (MG/DL)', 'HOMAIR']
categorical = ["GENDER", "RACE/HISPANIC ORIGIN W/ NH ASIAN", "isDM"]
continuous = [col for col in cols if col not in categorical]

# Usage example
TARGET = "isAtRiskMASH35"  # Example for one target

df = df[cols + [TARGET]]

# Impute
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

for col in categorical:
    df[col] = df[col].astype('category').cat.codes

analysis = StatisticalAnalysis(df[continuous + [TARGET]], TARGET)
continuous_summary = analysis.summarize_stats_continuous()
continuous_summary.to_csv("isAtRiskMASH35_continuous_vars.csv")

analysis = StatisticalAnalysis(df[categorical + [TARGET]], TARGET)
categorical_summary = analysis.summarize_stats_categorical()
categorical_summary.to_csv("isAtRiskMASH35_categorical_vars.csv")