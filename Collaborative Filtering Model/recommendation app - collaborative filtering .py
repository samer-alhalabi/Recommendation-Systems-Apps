#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
    
    # users who has seen the same movies
    #Filtering out users that have watched movies that the input has watched and storing it
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
    
    #Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
    userSubsetGroup = userSubset.groupby(['userId'])
    
    #Sorting it so users with movie most in common with the input will have priority
    # We will select a subset of users to iterate through. 
    # because we don't want to waste too much time going through every single user
    
    userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
    userSubsetGroup = userSubsetGroup[1:100]
    
    # Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, 
    # where the key is the user Id and the value is the coefficient
    
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    pearsonCorrelationDict = {}

    #For every user group in our subset
    for name, group in userSubsetGroup:
        #Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        #Get the N for the formula
        nRatings = len(group)
        #Get the review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        #And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        #Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        #Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)

        #If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[name] = 0


    # create df for results 
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    
    # get the top 50 users that are most similar to our targeted user
    topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    
    ##### rating of selected users to all movies #####
    
    # merge with ratings df to get rating for the top users
    topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

    #Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    
    #Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']    
    
    #Creates an empty dataframe
    recommendation_df = pd.DataFrame()
    #Now we take the weighted average
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    recommendation_df['movieId'] = tempTopUsersRating.index
    
    # sort and get top 20 movies that the algorithm recommand!
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

    recommendation_for_userid = movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20)['movieId'].tolist())][['title', 'genres']]
    
    recommendation_for_userid = recommendation_for_userid.reset_index()
    
    recommendation_for_userid.drop(columns="index", inplace=True)
    
    # print a text 
    print("Here are the top 20 movies we recommend for user {0} based on similarity with other users:".format(userid))

    return recommendation_for_userid


# In[11]:


recommender_top20(2000)


# In[ ]:




