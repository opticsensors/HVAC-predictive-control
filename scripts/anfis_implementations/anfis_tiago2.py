import tensorflow as tf

class ANFIS(tf.keras.Model):
    def __init__(self, n_inputs, n_rules, learning_rate=1e-2):
        super(ANFIS, self).__init__()
        self.n = n_inputs
        self.m = n_rules
        self.mu = tf.Variable(tf.random.normal([n_rules * n_inputs]), name="mu")
        self.sigma = tf.Variable(tf.random.normal([n_rules * n_inputs]), name="sigma")
        self.y = tf.Variable(tf.random.normal([1, n_rules]), name="y")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        # Compute rule activations
        rul = tf.reduce_prod(
            tf.reshape(
                tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(inputs, (1, self.m)), self.mu)) / tf.square(self.sigma)),
                (-1, self.m, self.n)),
            axis=2)

        # Fuzzy base expansion function
        num = tf.reduce_sum(tf.multiply(rul, self.y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(rul, axis=1), 1e-12, 1e12)
        out = tf.divide(num, den)
        return out

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.keras.losses.Huber()(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_true = data
        y_pred = self(x, training=False)
        loss = tf.keras.losses.Huber()(y_true, y_pred)
        return {'loss': loss}