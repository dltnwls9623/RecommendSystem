import time
from Dataset1.data_provider import DataSet as D
from Dataset1.model import MatrixFactorizationModel as MF


if __name__ == '__main__':
    dataset = D()
    x_train, y_train, x_test, y_test, num_users, num_movies = dataset.get_data2()
    model = MF(x_train, y_train, x_test, y_test, num_users, num_movies, k=5, alpha=0.01, beta=0.01, epochs=10, verbose=True)

    start = time.time()
    training_process = model.train()
    end = time.time()
    duration = end-start
    mm = duration // 60
    ss = duration % 60
    print("Training completed in %dmin %.4fsecs!!" % (mm, ss))

    print('Train error: %.4f' % training_process[-1][1])
    preds, test_error = model.test()
    print('Test error: %.4f' % test_error)
    filename = 'B_results_DS2'
    dataset.save_result(preds, filename=filename)


