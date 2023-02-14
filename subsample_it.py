import pandas as pd
from unicodedata import category
import logging


class SubSampler:
    def __init__(self, file_name, sample_text_col, intent_col):
        self.file_name = file_name
        self.sample_text_col = sample_text_col
        self.intent_col = intent_col

    
    def preprocess_name(self, text):
        return '_'.join(''.join(ch if not category(ch).startswith('P') or ch == '_' else ' ' for ch in text.lower().strip()).split())


    def read_file(self, file_name):
        df = pd.DataFrame()

        try:
            df = pd.read_excel(file_name)
        except Exception as e:
            logging.error(e)
            
            try:
                df = pd.read_csv(file_name)
            except Exception as e:
                logging.error(e)
        
        if not len(df):
            logging.warning("The file is empty or the file doesn't exist!")
        else:
            df.rename(columns = {
                self.sample_text_col: "sample_text",
                self.intent_col: "intent_name"},
                inplace = True)

            df["intent_name"] = df["intent_name"].apply(lambda x: self.preprocess_name(x))

        return df

    
    def find_token_cdf(self):
        print(self.read_file(self.file_name))


if __name__ == "__main__":
    ss = SubSampler(file_name = "training_data - 2023-02-13T152453.016.xlsx",
            sample_text_col = "Sample Text",
            intent_col = "Intent Name")

    print(ss.find_token_cdf())
