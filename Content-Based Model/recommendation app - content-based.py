#!/usr/bin/env python
# coding: utf-8

# In[3]:


# libarires

import pandas as pd
from math import sqrt
import numpy as np


def recommender_top20(userid):
    
    # create dataframes 
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    
    # Data manipulation - sperate year from title and make assign it to a column then clean up title
    movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
    movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    
    # split genre "|"
    movies_df['genres'] = movies_df.genres.str.split('|')

    # create movie with genere df
    moviesWithGenres_df = movies_df.copy()

    # one-hot enconding - iterate through the list of genres and place a 1 into the corresponding column
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1
    #Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)
    
    #Drop removes a specified row or column from a dataframe
    ratings_df = ratings_df.drop('timestamp', 1)

    # filtering on the target user from rating df
    user_df = ratings_df[ratings_df.userId == userid]
    
    # merge user_df with movies_df to get the movies info for the target user
    
    user_df = user_df.merge(movies_df, how='inner')
    
    # create user input movies - basically what did this user watch and the rating
    inputMovies = user_df[['movieId','title', 'rating']]
    
    # merge user input movies with movies w genere df - we will need this df to create user's weighted avg
    userGenreTable= inputMovies.merge(moviesWithGenres_df, how='inner').iloc[:, 5:]
    
    # create a user profile - weighted avg list of the user's input movies
    #Dot produt to get weights
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
    
    # get every movie in our original dataframe
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    # and drop the unnecessary information
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
    
    #Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
    
    #Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
    
    # create a df 
    recommendationTable = pd.DataFrame({'movieId' : recommendationTable_df.index,
                                  'rating': recommendationTable_df.values})

    #The final recommendation table (Top 20 Movies we recommand for targeted user)
    recommendation_for_userid = recommendationTable.merge(movies_df, how='inner').head(20).drop('rating', 1)
    recommendation_for_userid = recommendation_for_userid[["title", "genres"]]
    
    recommendation_for_userid = recommendation_for_userid.reset_index()
    
    recommendation_for_userid.drop(columns="index", inplace=True)
    
    
    # print a text 
    print("Here are the top 20 movies we recommend for user {0} based on what user has watched:".format(userid))

    return recommendation_for_userid


# In[4]:


recommender_top20(1988)


# In[ ]:




