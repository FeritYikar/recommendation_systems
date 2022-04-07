import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

st.set_page_config(layout="wide", )

st.image('what_to_watch.png')

spark = SparkSession\
        .builder\
        .appName('ALSExample').config('spark.driver.host', 'localhost')\
        .getOrCreate()
# Movie Recommender preparing
print_in = open('pickle/most_reviewed_movies_dict.pickle','rb')
most_reviewed_movies_dict = pickle.load(print_in)
print_in.close()

movie_ratings=spark.read.format('parquet').load('movie_ratings')

movie_titles=spark.read.format('parquet').load('movie_titles')

def name_retriever(movie_id, movie_title_df):
    return movie_title_df.where(movie_title_df.movieId == movie_id).take(1)[0]['title']

def new_user_recs(user_id, new_ratings, rating_df, movie_title_df, num_recs):
    # turn the new_recommendations list into a spark DataFrame
    new_user_ratings = spark.createDataFrame(new_ratings,rating_df.columns)
    
    # combine the new ratings df with the rating_df
    movie_ratings_combined = rating_df.union(new_user_ratings)
        
    # create an ALS model and fit it
    als = ALS(maxIter=5,rank=50, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
    model = als.fit(movie_ratings_combined)
    
    # make recommendations for all users using the recommendForAllUsers method
    recommendations = model.recommendForAllUsers(num_recs)
    
    # get recommendations specifically for the new user that has been added to the DataFrame
    recs_for_user = recommendations.where(recommendations.userId == user_id).take(1)
    
    user_recommendations = []
    for ranking, (movie_id, rating) in enumerate(recs_for_user[0]['recommendations']):
        movie_string = name_retriever(movie_id,movie_title_df)
        user_recommendations.append(movie_string)
        print('Recommendation {}: {}  | predicted score :{}'.format(ranking+1,movie_string,rating))
    return user_recommendations

# Book Recommender preparing


def main():
    st.title("Recommendations")
    menu = ['ratings', 'recommendations']
    st.sidebar.header('I need:')
    choice = st.sidebar.radio('', ('New Movies to Watch', 'New Books to Read'))

    if choice == 'New Movies to Watch':


        st.title("What to Watch")
        st.subheader('Rate the Movies You Have Watched')

        with st.form(key = 'New Movies to Watch'):
            ratings = {}
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                ratings[list(most_reviewed_movies_dict.keys())[0]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[0]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[1]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[1]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[2]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[2]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[3]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[3]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[4]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[4]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[5]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[5]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[6]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[6]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[7]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[7]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[8]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[8]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[9]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[9]}', ['Not seen', 1, 2, 3,4,5])
            with col2:
                ratings[list(most_reviewed_movies_dict.keys())[10]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[10]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[11]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[11]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[12]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[12]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[13]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[13]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[14]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[14]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[15]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[15]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[16]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[16]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[17]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[17]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[18]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[18]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[19]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[19]}', ['Not seen', 1, 2, 3,4,5])
            with col3:
                ratings[list(most_reviewed_movies_dict.keys())[20]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[20]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[21]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[21]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[22]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[22]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[23]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[23]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[24]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[24]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[25]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[25]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[26]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[26]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[27]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[27]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[28]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[28]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[29]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[29]}', ['Not seen', 1, 2, 3,4,5])
            with col4:
                ratings[list(most_reviewed_movies_dict.keys())[30]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[30]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[31]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[31]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[32]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[32]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[33]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[33]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[34]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[34]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[35]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[35]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[36]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[36]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[37]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[37]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[38]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[38]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[39]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[39]}', ['Not seen', 1, 2, 3,4,5])
            with col5:
                ratings[list(most_reviewed_movies_dict.keys())[40]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[40]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[41]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[41]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[42]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[42]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[43]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[43]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[44]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[44]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[45]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[45]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[46]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[46]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[47]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[47]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[48]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[48]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[49]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[49]}', ['Not seen', 1, 2, 3,4,5])
            no_recom = st.slider('How many movie recommendations do you want?', 0, 200, 5)
            submit_rates = st.form_submit_button(label='Submit')
        if submit_rates:
            user_ratings =[]
            for i in range(len(ratings)):
                if ratings[list(most_reviewed_movies_dict.keys())[i]] != 'Not seen':
                    item = (1000, list(most_reviewed_movies_dict.keys())[i], ratings[list(most_reviewed_movies_dict.keys())[i]])
                    user_ratings.append(item)
            user_recommendations = new_user_recs(1000,
                                                    new_ratings=user_ratings,
                                                    rating_df=movie_ratings,
                                                    movie_title_df=movie_titles,
                                                    num_recs = no_recom)
            st.write(user_recommendations[:no_recom])


    if choice == 'New Books to Read':


        st.title("What to Read")
        st.subheader('Rate the Books You Have Read')

        with st.form(key = 'New Movies to Watch'):
            ratings = {}
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                ratings[list(most_reviewed_movies_dict.keys())[0]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[0]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[1]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[1]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[2]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[2]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[3]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[3]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[4]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[4]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[5]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[5]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[6]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[6]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[7]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[7]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[8]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[8]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[9]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[9]}', ['Not seen', 1, 2, 3,4,5])
            with col2:
                ratings[list(most_reviewed_movies_dict.keys())[10]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[10]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[11]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[11]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[12]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[12]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[13]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[13]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[14]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[14]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[15]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[15]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[16]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[16]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[17]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[17]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[18]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[18]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[19]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[19]}', ['Not seen', 1, 2, 3,4,5])
            with col3:
                ratings[list(most_reviewed_movies_dict.keys())[20]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[20]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[21]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[21]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[22]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[22]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[23]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[23]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[24]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[24]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[25]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[25]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[26]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[26]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[27]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[27]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[28]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[28]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[29]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[29]}', ['Not seen', 1, 2, 3,4,5])
            with col4:
                ratings[list(most_reviewed_movies_dict.keys())[30]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[30]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[31]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[31]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[32]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[32]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[33]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[33]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[34]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[34]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[35]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[35]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[36]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[36]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[37]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[37]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[38]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[38]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[39]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[39]}', ['Not seen', 1, 2, 3,4,5])
            with col5:
                ratings[list(most_reviewed_movies_dict.keys())[40]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[40]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[41]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[41]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[42]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[42]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[43]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[43]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[44]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[44]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[45]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[45]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[46]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[46]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[47]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[47]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[48]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[48]}', ['Not seen', 1, 2, 3,4,5])
                ratings[list(most_reviewed_movies_dict.keys())[49]] = st.selectbox(f'{list(most_reviewed_movies_dict.values())[49]}', ['Not seen', 1, 2, 3,4,5])
            no_recom = st.slider('How many movie recommendations do you want?', 0, 200, 5)
            submit_rates = st.form_submit_button(label='Submit')
        if submit_rates:
            user_ratings =[]
            for i in range(len(ratings)):
                if ratings[list(most_reviewed_movies_dict.keys())[i]] != 'Not seen':
                    item = (1000, list(most_reviewed_movies_dict.keys())[i], ratings[list(most_reviewed_movies_dict.keys())[i]])
                    user_ratings.append(item)
            user_recommendations = new_user_recs(1000,
                                                    new_ratings=user_ratings,
                                                    rating_df=movie_ratings,
                                                    movie_title_df=movie_titles,
                                                    num_recs = no_recom)
            st.write(user_recommendations[:no_recom])






if __name__ =='__main__':
    main()











