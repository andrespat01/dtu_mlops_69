import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues

# To see the graph generated use:
# open report.html

# Getting the data ready:
reference_data_path = Path("data/raw/tweets.csv")
reference_data = pd.read_csv(reference_data_path)
reference_data = reference_data.drop(columns=['id', 'keyword'])
reference_data.columns = reference_data.columns.str.strip()

current_data = pd.read_csv('prediction_api/prediction_database.csv')
current_data = current_data.drop(columns=['time'])
current_data.columns = current_data.columns.str.strip()
current_data = current_data.rename(columns={"prediction": "target"})

# Generating the report:
report = Report(metrics=[DataDriftPreset(),
                DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')

# Testing the report:
data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result['summary']['all_passed'])
