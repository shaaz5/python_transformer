from transformers import pipeline
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

MyModel=pipeline(task="table-question-answering",model="google/tapas-large-finetuned-wtq")

MyDataset=pd.read_csv("C:\\My Doc\\python_pandas\\project\\addmission_info.csv")

MyDataset= MyDataset.astype(str)

question="how many ppl are under 18?"

print(MyModel(table=MyDataset,query=question)["answer"])

