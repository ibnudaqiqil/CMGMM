from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier

stream = SEAGenerator(random_state=1)

# Setup Adaptive Random Forest Classifier
arf = AdaptiveRandomForestClassifier()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 1200

# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = arf.predict(X)
    if y[0] == y_pred[0]:
        correct_cnt += 1
    arf.partial_fit(X, y)
    n_samples += 1

# Display results
print('Adaptive Random Forest ensemble classifier example')
print('{} samples analyzed.'.format(n_samples))
print('Accuracy: {}'.format(correct_cnt / n_samples))
