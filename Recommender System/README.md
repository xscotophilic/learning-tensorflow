# Recommender System

## Code Outline (Note: First go through all the theory):

## <!-- <img src="" alt="Outline"/> -->

## Theory section

---

<p align="center">
  <!-- <img src="" alt="RecommenderSystem"/> -->
</p>

- How to recommend?

  - suppose: User u1 gave Rick and Morty a 5, Bojack Horseman got a 5, and Wolfwalkers got a 5. User u1 is most likely a lover of Animation and Adventure.

  - Given a dataset of triples: (user, item, rating) Fit a model to the data: function(user, item) → rating.

  - What should it do? If the user u1 and the movie m1 were in the dataset, the projected rating should be close to the true rating. Even if movie m1 did not appear in the training set, the function should predict what user u1 would rank it.

  - Since our model can predict ratings for unseen movies, this is easy. Given a user, get predicted for every unseen movie, Sort by predicted rating (descending), Recommend movies with the highest predicted rating.

- The Dataset of Ratings Must Be Incomplete

  - There is nothing left to recommend if all users have seen all movies! Fortunately, this is not the case for real datasets.

  - <!-- <img src="" alt="Dataset"/> -->

- Flow diagram

  - <!-- <img src="" alt="Flowdiagram"/> -->
