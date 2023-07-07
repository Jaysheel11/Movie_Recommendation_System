# Movie_Recommendation_System
Recommender system using bag of words

The provided code combines a movie recommendation system with a Streamlit web application. Streamlit is a Python framework used for building interactive web applications for data science and machine learning projects.

The movie recommendation system is based on a trained model that utilizes preprocessed movie data and a similarity matrix. The code begins by importing the necessary libraries, including Streamlit, pickle, pandas, and requests. It defines functions for fetching movie posters from the TMDB API, recommending similar movies based on the trained model, and displaying the results in the Streamlit application.

The training of the recommendation model is not shown in the code provided. However, it typically involves several steps:

Loading and preprocessing the movie dataset: The movie dataset, such as 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv', is loaded using pandas. The data is then processed and cleaned to extract relevant features, such as genres, keywords, cast, crew, and overview.

Generating movie representations: The preprocessed data is transformed into numerical representations suitable for training the model. This may involve techniques such as one-hot encoding, vectorization, or embedding.

Building the recommendation model: A suitable model architecture, such as collaborative filtering, content-based filtering, or hybrid methods, is chosen and implemented. The model is trained using the movie representations and similarity scores calculated from the dataset.

Evaluating and fine-tuning the model: The trained model is evaluated using appropriate metrics to assess its performance. Fine-tuning techniques, such as hyperparameter tuning or cross-validation, may be applied to improve the model's accuracy and robustness.

Saving the model and similarity matrix: Once the model training is complete, the trained model and similarity matrix are typically saved using pickle or a similar serialization method. This allows for later retrieval and use in the recommendation system.

Requirements:
To run the complete movie recommendation system, ensure the following requirements are met:

The preprocessed movie dataset, including 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv', is available.
The model training code, which is not provided, should be executed separately to train and save the recommendation model and similarity matrix.
The trained model and similarity matrix should be saved as 'movie_dict.pkl' and 'similarity.pkl', respectively.
Install the necessary libraries using !pip install streamlit pickle pandas requests.
By running the Streamlit application, users can select a movie from the list and receive personalized movie recommendations along with the movie posters. The recommendation system utilizes a trained model and precomputed similarity scores to provide relevant suggestions based on the selected movie.
