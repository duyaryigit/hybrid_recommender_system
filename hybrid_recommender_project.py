#############################################
# PROJECT: Hybrid Recommender System
#############################################

# Make a guess for the user whose ID is given, using the item-based and user-based recommender methods.
# Consider 5 suggestions from the user-based model and 5 suggestions from the item-based model and finally make 10 suggestions from 2 models.

#############################################
# Task 1: Prepare and Understand Data (Data Understanding)
#############################################

import warnings
import pandas as pd
warnings.filterwarnings("ignore")
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# 1: Read the Movie and Rating datasets. dataset containing
# movieId, movie name and genre information of the movie

movie = pd.read_csv('datasets/movie.csv)
movie.head()
movie.shape

# Dataset containing UserID, movie name, movie vote and time information

rating = pd.read_csv('datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# 2: Add the names and genre of the movies to the rating dataset using the movie dataset.
# Only the id's of the movies that the users voted in the rating have.
# We add the movie names and genre of the ids from the movie dataset.

df = movie.merge(rating,how="left", on="movieId")
df.head()
df.shape
df["rating"].isnull().sum()

"""
#################
# Title
#################

# If we use regex to extract the year in the title;

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)  # Capturing expressions with 4 values in them
df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False) # Take inside parentheses
df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')                 # Clearing year inside title
df['title'] = df['title'].apply(lambda x: x.strip())                     # Deleting spaces in the original title

df.shape
df.head()

#################
# Genres
#################

# If we want to take the first expression of genres and discard the others:

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()

#################
# Timestamp
#################

df.info() # The timestamp variable is in object format, it needs to be converted to datetime:

df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')   # We expressed the format as year-month-day so that we can get separately.
df.info()

df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
"""

# 3: Calculate the total number of people who voted for each movie. Remove the movies with less than 1000 votes from the data set.
# We calculate how many people voted in total for each movie.

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

# Keeping the names of movies with less than 1000 votes in rare_movies and remove them from the data set.

rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies.head()


# 4: Create a pivot table for the dataframe with userId's, movie names in the columns and ratings as values.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# 5: Let's functionalize all the above operations:

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


##### *** ADDITIONAL INFO *** ###
# Dump and store when the data is large so we don't call it repeatedly, calling the USER-MOVIE DF for later use:

import pickle
pickle.dump(user_movie_df, open("user_movie_df.pkl", "wb"))
user_movie_df = pickle.load(open("user_movie_df.pkl", "rb"))

#############################################
# Task 2: Determining the Movies Watched by the User to suggest
#############################################

# 1: Choose a random user id.

random_user = 108170

# 2: Create a new dataframe named random_user_df consisting of observation units of the selected user.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# 3: Assign the movies voted by the selected user to a list called movies_watched.

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched
movie.columns[movie.notna().any()].to_list()

#############################################
# Task 3: Accessing Data and Ids of Other Users Watching the Same Movies
#############################################

# 1: Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Step 2: Create a new dataframe named user_movie_count, which contains the information of how many movies each user watched, and we create a new df.

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count.head()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# 3: We consider those who watch 60 percent or more of the movies voted by the selected user as similar users.
# Create a list named users_same_movies from the ids of these users.

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)

#############################################
# Task 4: Identifying Users Most Similar to the User to Suggest
#############################################

# 1: Filter the movies_watched_df dataframe to find the ids of the users that are similar to the selected user in the user_same_movies list.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# 2: Create a new corr_df dataframe where users' correlations with each other will be found.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df.index
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.head()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# 3: Create a new dataframe named top_users by filtering out users that have high correlation (over 0.65) with the selected user.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users

# 4: Merge the top_users dataframe with the rating dataset

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.tail()
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()

#############################################
# Task 5: Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies
#############################################

# Step 1: Create a new variable named weighted_rating, which is the product of each user's corr and rating.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# 2: Create a new dataframe named recommendation_df containing the movie id and the average value of the weighted ratings of all users for each movie.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# 3: Step3: Select the films with a weighted rating greater than 3.5 in recommendation_df and rank them by weighted rating.
# Save the first 5 observations as movies_to_be_recommend.

recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# 4: Bring the names of the 5 recommended movies.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

'''
# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)
'''

#############################################
# Task 6: Item-Based Recommendation
#############################################

# Make an item-based suggestion based on the name of the movie that the user last watched and gave the highest rating.

user = 108170

# 1: Read movie and rating datasets.

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# 2: Get the id of the movie with the most recent score from the movies that the user to be recommended gives 5 points.

movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# 3: Filter the user_movie_df dataframe created in the User based recommendation section according to the selected movie id.

movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# 4: Find the correlation of the selected movie with the other movies using the filtered dataframe and rank them.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Function that performs the last two steps:

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# 5: Give the first 5 movies as suggestions except the selected movie itself.

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# 1 to 6th. 0th index has the movie itself. We left it out.

movies_from_item_based[1:6].index.to_list()

'''
# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']
'''

hybrid=pd.DataFrame()
hybrid["user_based_rec"]=movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]
hybrid["item_based_rec"]=movies_from_item_based[1:6].index.to_list()
