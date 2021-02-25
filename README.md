## Recommendation Systems

Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous, and can be commonly seen in online stores, movies databases and job finders. In this repo, I will explore two systems: **Content-based** & **Collaborative Filtering** systems.

### Content-Based recommendation system aka (Item-Item recommendation):
This technique attempts to figure out what a user's favourite aspects of an item is, and then recommends items that present those aspects. In my case, I'm going to try to figure out the input's favorite genres from the movies and ratings given.

### Collaborative Filtering aka (User-User Filtering):
This technique uses other users to recommend items to the input user. It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input. There are several methods of finding similar users (Even some making use of Machine Learning), and the one we will be using here is going to be based on the Pearson Correlation Function.

![image](images/Collaborative-Filtering.jpg)
