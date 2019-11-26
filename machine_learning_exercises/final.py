import NNfactory
from time import sleep
import sys
import numpy as np
import matplotlib.pyplot as plt


# train

def train(mynn, data_type, epochs, n_data_max, dn_data, training_data_list, test_data_list, ):
    train_results = list()
    test_results = list()
    for e in range(epochs):
        # go through all records in the training data set
        print(' * epoch = {}'.format(e + 1))
        id_data = 0

        for record in training_data_list[:n_data_max]:

            id_data += 1
            if (id_data % dn_data == 0):
                sys.stdout.write('\r')
                sys.stdout.write(' [%-20s] %d%%' % ('=' * (id_data // dn_data), 5 * (id_data // dn_data)))
                sys.stdout.flush()
                sleep(0.25)

            # split the record by the ',' commas
            all_values = record.split(',')

            # 입력데이터 ['x1','x2',...] & 입력데이터의 스케일링 (n_input_nodes, )
            if data_type == 'mnist':
                input_list = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            else:
                input_list = np.asfarray(all_values[1:])

            # create the target output values (shape = (10,))
            target_list = np.zeros(10)  # mynn.n_nodes[-1])

            # all_values[0] is the target label for this record
            target_list[int(all_values[0])] = 1.0

            mynn.train(input_list, target_list)

            pass

        print('')
        print(' > 훈련 샘플에 대한 성능 (정확도 & 평균에러) ')
        train_result = mynn.check_accuracy_error(
            training_data_list, 0, n_data_max, data_type='mnist')
        print('')
        print(' > 테스트 샘플에 대한 성능 (정확도 & 평균에러) ')
        test_result = mynn.check_accuracy_error(test_data_list, 0, len(test_data_list), data_type='mnist')
        train_results.append(train_result)
        test_results.append(test_result)
        pass
    return train_results, test_results


def draw_result(training_results, test_results):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim([1, len(training_results) + 1])
    x = np.linspace(1, len(training_results), len(training_results))
    ax.plot(x, [x['accuracy'] for x in training_results], 'bo-', label='training accuracy')
    ax.plot(x, [x['accuracy'] for x in test_results], 'go-', label='test accuracy')
    ax.set_xticks(np.arange(0, len(training_results) + 1))
    plt.show()


# 데이터 불러오기 (MNIST 손글씨 데이터)
def run():
    # 훈련데이터
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 테스트데이터
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    print('training data size : {}'.format(len(training_data_list)))
    print('test data size : {}'.format(len(test_data_list)))

    name = 'myMLP'
    structure = '784:identity|100:relu|100:tanh|10:softmax'
    mynn = NNfactory.MLP(model_structure=structure, model_nametag=name, learning_rate=0.005)

    # 네트웍의 훈련: the neural network
    data_type = 'mnist'

    # 최대 학습 주기 설정
    epochs = 3

    n_data_tot = len(training_data_list)
    n_data_max = 60000  # 훈련에 사용할 데이터 갯수 (최대 60000)
    n_data_test = min(int((n_data_tot - n_data_max) * 0.7), int(n_data_max * 0.7))
    n_data = len(training_data_list[:n_data_max])
    dn_data = int(n_data / 20)
    print(dn_data)

    train_results, test_results = train(mynn, data_type, epochs, n_data_max, dn_data, training_data_list,
                                        test_data_list)
    draw_result(train_results, test_results)
    mynn.save_model(fname='mymodel.npy', nametag='first run')

    mymodel = np.load('mymodel.npy')
    mynn2 = NNfactory.MLP(load_model_np=mymodel)

    data = training_data_list[0].split(',')
    data[0]
    input_list = np.asfarray(data[1:]) / 255 * 0.99 + 0.01

    # 로드된 모형(mynn2)의 순방향 출력 체크 (첫번째 데이터에 대하여)
    np.argmax(mynn2.feedforward(input_list))


if __name__ == '__main__':
    run()
