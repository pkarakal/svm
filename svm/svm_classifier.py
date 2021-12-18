import os
import time
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from svm.utils import download_mnist_dataset, split_into_even_and_odd
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale


def svm_classifier(train_set, train_labels, test_set, test_labels):
    start_time = time.time()
    grid_params = {
        'gamma': [1e-2, 1e-3, 1e-4],
        'C': [5, 10]
    }
    X_train = scale(train_set)
    X_test = scale(test_set)
    svc = SVC(kernel='rbf', shrinking=True, cache_size=1000, random_state=42)
    grid = GridSearchCV(svc, grid_params, refit=True, n_jobs=-1, verbose=5)
    grid.fit(X_train, train_labels)
    print(f"Training Time: {(time.time() - start_time)} seconds.")
    start_time = time.time()
    pred = grid.predict(X_test)
    print(f"Prediction Time: {(time.time() - start_time)} seconds.")
    clf = grid.best_estimator_
    print(f"Best estimator was {grid.best_params_}")
    start_time = time.time()
    clf.score(X_test, test_labels)
    print(f"Prediction time for best estimator: {(time.time() - start_time)} seconds.")
    print(pd.DataFrame(grid.cv_results_))
    report_nc = classification_report(test_labels, pred)
    print(report_nc)


def main():
    train_dataset = download_mnist_dataset('mnist', True, transform=None)
    split_into_even_and_odd(train_dataset)
    train_df = train_dataset.data.numpy()
    train_label = train_dataset.targets.numpy()

    train_df = np.reshape(train_df, (train_df.shape[0], train_df.shape[1] * train_df.shape[2]))

    test_dataset = download_mnist_dataset('mnist', False, transform=None)
    split_into_even_and_odd(test_dataset)
    test_df = test_dataset.data.numpy()
    test_label = test_dataset.targets.numpy()

    test_df = np.reshape(test_df, (test_df.shape[0], test_df.shape[1] * test_df.shape[2]))
    svm_classifier(train_df, train_label, test_df, test_label)


if __name__ == '__main__':
    main()
