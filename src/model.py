import tensorflow as tf
from tensorflow.keras import layers, Model

class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name="user_embedding"
        )
        self.user_bias = layers.Embedding(num_users, 1, name="user_bias")
        
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            name="movie_embedding"
        )
        self.movie_bias = layers.Embedding(num_movies, 1, name="movie_bias")
        
    def call(self, inputs):
        # inputs shape: (batch_size, 2) -> [user_index, movie_index]
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        # Dot product of user and movie embeddings
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1, keepdims=True)
        
        # Add biases
        x = dot_user_movie + user_bias + movie_bias
        
        # Scale output to 0-1 then to 0-5.5 (to cover 0.5-5.0 range comfortably)
        return tf.nn.sigmoid(x) * 5.5
