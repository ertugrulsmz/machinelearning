class CustomFit(keras.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.layer1 = layers.Conv2D(64, (3, 3), padding="same",activation='relu')
        self.layer2 = layers.Conv2D(128, (3, 3), padding="same",activation = 'relu')
        self.layer3 = layers.Flatten()
        self.layer4 = layers.Dense(10)

    def call(self,inputs,training):

      # if training == True # not important
      x = self.layer1(inputs)
      x = self.layer2(x)
      x = self.layer3(x)
      return self.layer4(x)


    def compile(self, optimizer, loss):
        super(CustomFit, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Caclulate predictions
            y_pred = self.call(x, training=True)

            # Loss
            loss = self.loss(y, y_pred)

        # Gradients
        training_vars = self.trainable_variables
        gradients = tape.gradient(loss, training_vars)

        # Step with optimizer
        self.optimizer.apply_gradients(zip(gradients, training_vars))
        acc_metric.update_state(y, y_pred)

        return {"loss": loss, "accuracy": acc_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_pred = self.call(x, training=False)

        # Updates the metrics tracking the loss
        loss = self.loss(y, y_pred)

        # Update the metrics.
        acc_metric.update_state(y, y_pred)
        return {"loss": loss, "accuracy": acc_metric.result()}


acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

training = CustomFit(model)
training.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

training.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=64, epochs=2)
