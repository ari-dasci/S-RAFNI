import tensorflow as tf

class CustomScheduleC10(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        if step > 25040: #31280:
            lr = 0.001
        elif step > 12520: #15640:
            lr = 0.01
        else:
            lr = 0.1
        return lr

class CustomScheduleC100(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        if step > 46920:
            lr = 0.001
        elif step > 31280:
            lr = 0.01
        else:
            lr = 0.1
        return lr
