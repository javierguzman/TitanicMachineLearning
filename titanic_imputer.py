# -*- coding: utf-8 -*-

import numpy
import pandas 

from sklearn.base import TransformerMixin

class TitanicImputer(TransformerMixin):

    def __init__(self):
        """Impute missing embarked values.
        """
    def fit(self, X, y=None):        
        self.fill = X.value_counts().index[0]
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)