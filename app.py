import streamlit as st

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats, optimize, interpolate

@dataclass
class Distribution:
    sex: str
    age: float
    
    mean: float
    std: float
    
    givens: dict
        
    def to_z_score(self, num):
        return (num-self.mean)/self.std
    
    def to_percentile(self, num):
        
        z_score = self.to_z_score(num)
        return 1 - stats.norm.sf(z_score)
    
    def __hash__(self):
        return f'{self.age, self.sex, self.mean, self.std}'
        
def load_data(file='zwtageinf.xls'):
    return pd.read_excel(file).loc[lambda x: x['Sex'] != 'Sex'].astype(float)

def add_null_rows(df):
    nans = np.where(np.empty_like(df.values), np.nan, np.nan)
    data = np.hstack([nans, df.values]).reshape(-1, df.shape[1])
    return pd.DataFrame(data, columns=df.columns)
    
def interpolate_values(df):
    return df.interpolate(method='polynomial', order=1)

def row_to_dist(x):
    return Distribution(sex=x['Sex'], 
                        age=x['Agemos'], 
                        mean=x[0], 
                        std=x[1]-x[0], givens=x.to_dict())

def add_dist(df):
    return df.assign(Dist = lambda x: x.apply(lambda row: row_to_dist(row), axis=1))

@st.cache_data
def init_data():
    df = (
        load_data()
        .pipe(add_null_rows).pipe(interpolate_values).dropna()
        .pipe(add_null_rows).pipe(interpolate_values).dropna()
        .replace(to_replace={'Sex':{1.0:'Male', 2.0:'Female'}})
        .pipe(add_dist)
        .set_index(['Sex', 'Agemos'])
    )
    return df

data = init_data()

st.title('Is My Baby Heavy?')

age, weight, sex = st.columns((1,1,1))

with age:
    baby_age = st.number_input('Baby age in months', min_value=0.0, max_value=36.0, step=0.25, value=2.0)

with weight:
    baby_weight = st.number_input('Baby weight in KG', min_value=0.0, max_value=12.0, step=0.01, value=6.0)

with sex:
    baby_sex = st.selectbox('Baby sex', ['Male', 'Female'])

dist = data.loc[(baby_sex, baby_age), 'Dist']

st.success(f'{dist.to_percentile(baby_weight):.2%}')

x = data.loc[lambda x: x.index.get_level_values('Sex') == baby_sex].loc[lambda x: x['Dist'].apply(lambda x: x.mean) >= baby_weight].index[0][1]



st.success(f'Your baby is heavier than an average {x} month old!')
