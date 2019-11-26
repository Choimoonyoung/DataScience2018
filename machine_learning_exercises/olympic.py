import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import sys
import scipy.special

# 데이터 불러오기 (MNIST 손글씨 데이터)
training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

all_values = training_data_list[0].split(',')


image_array = np.asfarray(all_values[1:]).reshape((28,28))
# print (image_array)

plt.imshow(image_array, cmap='Greys')

sfactor = 4
nx, ny = 5, 5
figsize = tuple(sfactor * np.array((nx, ny)))

fig = plt.figure(figsize=figsize)
ax = [fig.add_subplot(ny, nx, i + 1) for i in range(nx * ny)]

for i in range(nx * ny):
    all_values = training_data_list[i].split(',')
#     all_values = training_data_list[np.random.randintimport numpy as np


class NeuralNetwork:

    def __init__(self, n_inodes, n_hnodes, n_onodes, learning_rate):
        #         pass
        # 입력층, 은닉층, 출력층 노드 개수의 설정
        self.inodes = n_inodes
        self.hnodes = n_hnodes
        self.onodes = n_onodes

        # 가중치 행렬
        self.w_hi = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_oh = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 학습률
        self.learning_rate = learning_rate

        # 활성화 함수 (activation function : sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def softmax(self, x):
        """
        * softmax actvation function
        """
        return np.exp(x) / np.sum(np.exp(x))

    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원의 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 은닉층으로 들어오는 신호를 계산
        hidden_inputs = np.matmul(self.w_hi, inputs)

        # 은닉층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력층으로 들어오는 신호를 계산
        final_inputs = np.matmul(self.w_oh, hidden_outputs)

        # 최종 출력층에서 나가는 신호를 계산
        #         final_outputs = self.activation_function(final_inputs)
        final_outputs = self.softmax(final_inputs)

        # 출력층의 오차는 (실제값 - 계산값)
        output_errors = targets - final_outputs

        # 은닉층의 오차는 가중치에 의해 나뉜 출력계층의 오차들을 재조합해 계산
        hidden_errors = np.matmul(self.w_oh.T, output_errors)

        # 은닉층과 출력층간의 가중치 업데이트
        self.w_oh += self.learning_rate * np.matmul((output_errors * final_outputs * (1.0 - final_outputs)),
                                                    np.transpose(hidden_outputs))

        # 입력층과 은닉층간의 가중치 업데이트
        self.w_hi += self.learning_rate * np.matmul((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                    np.transpose(inputs))

        pass

    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T

        # 은닉층으로 들어오는 신호를 계산
        hidden_inputs = np.matmul(self.w_hi, inputs)

        # 은닉층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)

        # 출력층으로 들어오는 신호를 계산
        final_inputs = np.matmul(self.w_oh, hidden_outputs)

        # 출력층에서 나가는 신호를 계산
        #         final_outputs = self.activation_function(final_inputs)
        final_outputs = self.softmax(final_inputs)

        return final_outputs

    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    ax[i].imshow(image_array, cmap='Greys', interpolation='None')
    ax[i].grid(True)


fig.tight_layout()

plt.show()

# 입력(input), 은닉(hidden), 출력(output)층(layer) 노드의 수

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# 학습률 (learning rate)
learning_rate = 0.3

# 신경망 클래스의 인스턴스 생성
mynn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 데이터 불러오기 (MNIST 손글씨 데이터)

training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 네트웍의 훈련: the neural network

# 최대 학습 주기 설정
epochs = 3

n_data_max = 60000  # 훈련에 사용할 데이터 갯수 (최대 60000)
n_data = len(training_data_list[:n_data_max])
dn_data = int(n_data / 20)
print(dn_data)

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
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes)

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 1.0

        mynn.train(inputs, targets)
        pass

    print('\n')

    pass

# 테스트 데이터 로딩

test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 테스트 데이터를 사용한 훈련한 신경망의 테스트

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     inputs = (np.asfarray(all_values[1:]) / 255.0 * 1.0) + 0.0

    # query the network
    outputs = mynn.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

nid=1
all_values=test_data_list[nid].split(',')
print(all_values[0])
image_array=np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
mynn.query((np.asfarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01)

mynn.query((np.asfarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01)