# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:13:47 2018

@author: 20767
"""


import os
import pandas as pd

dataset = pd.read_csv(os.getcwd()+'\\data\\movies_metadata.csv', low_memory=False)

"""
Broadly, recommender systems can be classified into 3 types:

Simple recommenders: offer generalized recommendations to every user, based on movie popularity and/or genre. 
                     The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience. 
                     IMDB Top 250 is an example of this system.
Content-based recommenders: suggest similar items based on a particular item. 
                            This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations.
                            The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
Collaborative filtering engines: these systems try to predict the rating or preference that a user would give an item-based on past ratings and preferences of other users. 
                    Collaborative filters do not require item metadata like its content-based counterparts.

"""


"""
    Simple Recommendation 
    
The following are the steps involved:

Decide on the metric or score to rate movies on.
Calculate the score for every movie.
Sort the movies based on the score and output the top results.
"""
C = dataset['vote_average'].mean()
m = dataset['vote_count'].quantile(0.90)
new_dataset = dataset.copy().loc[dataset['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

new_dataset['score'] = new_dataset.apply(weighted_rating, axis=1)
new_dataset = new_dataset.sort_values('score',ascending=False)

print (new_dataset['title'].head(10))


