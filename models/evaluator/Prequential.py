"""Prequential data stream evaluator."""

import numpy as np

from sklearn.metrics import accuracy_score

from ..metrics import balanced_accuracy_score


class Prequential:


    def __init__(self, metrics=(accuracy_score, balanced_accuracy_score)):
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]

    def process(self, stream, clfs, interval=100):

        # Assign parameters
        self.stream_ = stream
        self.interval_ = interval

        intervals_per_chunk = int(self.stream_.chunk_size / self.interval_)
        self.scores = np.zeros(
            (
                len(self.clfs),
                ((stream.n_chunks - 1) * intervals_per_chunk),
                len(self.metrics),
            )
        )

        i = 0
        while True:
            stream.get_chunk()
            a, _ = stream.current_chunk
            # break

            if stream.previous_chunk is not None:
                X_p, y_p = stream.previous_chunk
                X_c, y_c = stream.current_chunk

                X = np.concatenate((X_p, X_c), axis=0)
                y = np.concatenate((y_p, y_c), axis=0)

                for interval_id in range(1, intervals_per_chunk + 1):
                    start = interval_id * interval
                    end = start + self.stream_.chunk_size

                    for clfid, clf in enumerate(self.clfs):
                        y_pred = clf.predict(X[start:end])

                        self.scores[clfid, i] = [
                            metric(y[start:end], y_pred) for metric in self.metrics
                        ]

                    [clf.partial_fit(X[start:end], y[start:end])
                     for clf in self.clfs]

                    i += 1
            else:
                X_train, y_train = stream.current_chunk
                [
                    clf.partial_fit(X_train, y_train, self.stream_.classes_)
                    for clf in self.clfs
                ]

            if stream.is_dry():
                break