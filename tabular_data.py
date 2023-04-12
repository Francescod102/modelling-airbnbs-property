
# Import libraries
from ast import literal_eval
import numpy as np
import pandas as pd

# Define a function which removes missing values in the rating columns
def remove_rows_with_missing_ratings(df) -> pd.DataFrame:
    df = df.dropna(subset = ["Cleanliness_rating",
                            "Accuracy_rating",
                            "Communication_rating",
                            "Location_rating",
                            "Check-in_rating",
                            "Value_rating"])
    return df

# Define a function that combines the list items into the same string
def combine_description_strings(df) -> pd.DataFrame:
    
    def convert_description_strings(x: str):
        """
        Convert a list of strings to a single string by
        concatenating the remaining strings
        
        """
        try:
            x = literal_eval(x)
            x.remove("About this space")
            x = "".join(x)
            return x
        except:
            print(f"Failed to convert string: {x}")
            return np.nan
    
    df["Description"] = df["Description"].dropna().apply(convert_description_strings)
    df["Description"].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex = True, inplace = True)
    return df

# Define a function to fill missing values with 1 in specific columns
def set_default_feature_values(df) -> pd.DataFrame:
    cols_to_fill = ["guests", "beds", "bathrooms", "bedrooms"]
    df[cols_to_fill] = df[cols_to_fill].apply(pd.to_numeric, errors = "coerce").fillna(1)
    return df

# Define a function that returns the features and labels of your data in a tuple
def load_data(df, labels) -> pd.DataFrame:
    features = df.drop(columns = labels, axis = 1)
    labels = df[labels]
    return features, labels

# Encapsulate all the processing code inside a function
def clean_tabular_data(df) -> pd.DataFrame:
    """
    Clean tabular data by removing rows with missing ratings,
    combining description strings, and setting default feature values
    """
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)

    """Return a dataframe contains only columns with numerical data types"""
    df.drop("Unnamed: 19", axis = 1, inplace = True)

    return df

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    try:
        raw_data = pd.read_csv("../data/raw_tabular_data.csv", index_col = "ID")
    except:
        print("Failed to read input file")
    df = clean_tabular_data(raw_data)
    try:
        df.to_csv("../data/clean_tabular_data.csv")
    except:
        print("Failed to write output file")