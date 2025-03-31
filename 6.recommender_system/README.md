# Recommender System

## How to Recommend?

- Suppose: User **u1** rated *Rick and Morty* a **5**, *Bojack Horseman* a **5**, and *Wolfwalkers* a **5**. This suggests that user **u1** is likely a fan of **Animation** and **Adventure**.
- Given a dataset of triples: **(user, item, rating)**, we can fit a model to the data such that:
  **function(user, item) → rating**
- What should the model do?
  - If **user u1** and **movie m1** exist in the dataset, the predicted rating should be close to the actual rating.
  - Even if **movie m1** was not in the training set, the model should still predict how **user u1** would rate it.
- Since our model can predict ratings for unseen movies, the recommendation process is straightforward:
  1. Given a user, predict ratings for every unseen movie.
  2. Sort the movies by predicted rating (descending order).
  3. Recommend movies with the highest predicted ratings.

## The Dataset of Ratings Must Be Incomplete

- If all users have rated all movies, there would be nothing left to recommend! Fortunately, real-world datasets are incomplete, making recommendations possible.
  - <img src="https://user-images.githubusercontent.com/47301282/119214359-dfcc9000-bae3-11eb-8048-58036222d14c.png" alt="Dataset"/>

---

## Flow Diagram for Data

- <img src="https://user-images.githubusercontent.com/47301282/119214730-9762a180-bae6-11eb-9ad1-e9689ac6215f.jpg" alt="Flowdiagram"/>
