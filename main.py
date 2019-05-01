import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import csv
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn import svm
import random
import time
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow import keras
import tensorflow

class MFCC_Generator(object):

    def __init__(self):

        self.csv_reader = CSVReader(TF_Trainer.time_data_csv_file)
        self.calculated_coeffs = {}

    def calculate_mfc_coeffs(self, sound_file, normalize=True):

        if sound_file in self.calculated_coeffs:
            return self.calculated_coeffs[sound_file]

        # Constants
        pre_emphasis = .97
        frame_stride = 0.01  # 0.01     # 10 ms overlap of dft windows
        frame_size = .025  # 0.025      # 25 ms for dft window size
        NFFT = 512  # FFT bin size
        nfilt = 40  # Number of mel filters
        num_ceps = 12
        sample_time = 3.5  # how long to read from audio
        cep_lifter = 22

        # ae_wav_file = "vowels/men/m01ae.wav"
        start_end_t = self.csv_reader.get_start_end_time_from_fname(sound_file)

        sample_rate, signal = scipy.io.wavfile.read(sound_file)
        sample_rate = float(sample_rate)

        signal = signal[ int( start_end_t[0] * sample_rate) : int(start_end_t[1] * sample_rate) ]  # Keep the first 3.5 seconds

        # y(t) = x(t) - pre_emphasis*x(t-1)
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(
            float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal,
                               z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        # See http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ for below
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        # Apply Discrete Cosine Transformation
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-num_ceps+1
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

        if normalize:
            mfcc /= np.max(np.abs(mfcc))
        # filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

        # print "  sample rate:", sample_rate
        # print "  frame_size:", frame_size
        # print "  frame_length:", frame_length
        # print "  num_frames:", num_frames
        # print "  filter banks shape:", filter_banks.shape
        # print "  mfcc shape:", mfcc.shape

        # plt.figure(1)
        # plt.subplot(211)
        # plt.imshow(filter_banks.T, cmap=plt.cm.jet, aspect='auto')
        # plt.title('normalized filter banks')
        # plt.subplot(212)
        # plt.imshow(mfcc.T, cmap=plt.cm.jet, aspect='auto')
        # plt.title('normalized mfcc coeffs')
        # plt.show()

        # if plot_d_emphasize_t_domain:
        #     plt.plot(signal)
        #     plt.xlabel('time (s)')
        #     plt.ylabel('signal amplitude')
        #     plt.title('Time vs Amplitude of Original Signal')
        #     plt.grid(True)
        #     # plt.savefig("test.png")
        #     plt.ion()
        #     plt.draw()
        #
        #
        # if plot_t_domain:
        #     plt.plot(signal)
        #     plt.xlabel('time (s)')
        #     plt.ylabel('signal amplitude')
        #     plt.title('Time vs Amplitude of Original Signal')
        #     plt.grid(True)
        #     # plt.savefig("test.png")
        #     plt.ion()
        #     plt.draw()
        #
        # scipy.io.wavfile.write('test.wav', sample_rate, signal)

        # plt.show()

        self.calculated_coeffs[sound_file] = mfcc
        return mfcc

class CSVReader(object):

    def __init__(self, time_file):

        self.time_file = time_file
        self.times = {}
        self.save_times_to_dict()

    def save_times_to_dict(self):

        with open(self.time_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.times[row[0]] = [ 0.001*float(row[1]), 0.001* float(row[2]) ] # multiply be 1000 to convert to seconds


    def sound_name_from_file(self, fname):
        _ = fname.split("/")
        return _[len(_)-1].split(".")[0]

    def get_start_end_time_from_fname(self, f_name):
        if len(self.times) == 0:
            return
        sound_name = self.sound_name_from_file(f_name)
        return self.times[sound_name]

class TF_Trainer(object):

    time_data_csv_file = 'vowels/timedata.csv'

    # ae
    # ah
    # aw
    # eh
    # er
    # ih
    # iy
    # oa
    # oo
    # uh
    # uw

    # vowels = [ "ae", "ah", "aw", "eh", "ei", "er", "ih", "iy", "oa", "oo", "uh", "uw" ]

    def __init__(self, training_vowels):

        self.train_data = []
        self.test_data = []
        self.train_classifications = []
        self.test_classifications = []

        self.training_vowels = training_vowels
        self.training_vowels_accuracy = [] # [[vowel 1 correct cnt, vowel 1 err count],[vowel2 correct cnt, vowel2 err count],..]

        self.mfcc_generator = MFCC_Generator()

    def get_vowels(self):
        return self.training_vowels

    def tf_format_mfccoeffs(self, coeffs, coeff_vectors_to_include=6, interprolate_coeffs=False, interprolated_array_len=6, debug=False):

        '''
        :summary If interprolate_coeffs is False, the first coeff_vectors_to_include vectors from coeffs are returned as
                 a numpy array.
                 If interprolate_coeffs is True, the function returns an array of length interprolated_array_len vectors
                 which are linearly interprolated from coeffs
        :param coeffs:
        :param coeff_vectors_to_include:
        :param interprolate_coeffs:
        :param debug:
        :return:
        '''

        if not interprolate_coeffs:
            ret = []
            for i in range(coeff_vectors_to_include):
                ret.append(coeffs[i])
            return np.array(ret)

        t = np.linspace(0, 10, len(coeffs))
        t_new = np.linspace(0, 10, interprolated_array_len)
        f = interp1d(t, coeffs, axis=0)
        coeffs_new = f(t_new)
        return coeffs_new

    def train_predict(self, pct_to_train=.8, coeff_vectors_to_include=15, interprolate_coeffs=False, interprolated_array_len=15, normalize_mfccs=True, debug=False, plot_epoch_acc=False, epochs=15):

        directory = "vowels/audioclips"

        for sound_name in os.listdir(directory):
            sound_name_no_wav = sound_name.split('.')[0]
            file_name = directory + "/" + sound_name
            for vowel in self.training_vowels:
                if vowel in sound_name_no_wav:

                    ret = self.mfcc_generator.calculate_mfc_coeffs(file_name, normalize=normalize_mfccs)
                    data = self.tf_format_mfccoeffs(ret, coeff_vectors_to_include=coeff_vectors_to_include, interprolate_coeffs=interprolate_coeffs, interprolated_array_len=interprolated_array_len)

                    classifier_num = self.training_vowels.index(vowel)

                    # TODO: Randomly sample from files, dont use >
                    if int(sound_name_no_wav[1:3]) > (1.0 - pct_to_train) * 50:
                        self.train_data.append(data)
                        self.train_classifications.append(classifier_num)
                    else:
                        self.test_data.append(data)
                        self.test_classifications.append(classifier_num)

        test_data = np.array(self.test_data)
        training_data = np.array(self.train_data)
        num_classes = len(self.training_vowels)
        if not interprolate_coeffs:
            test_data = test_data.reshape(test_data.shape[0], coeff_vectors_to_include, 12, 1)
            training_data = training_data.reshape(training_data.shape[0], coeff_vectors_to_include, 12, 1)
            input_shape = (coeff_vectors_to_include, 12, 1)
        else:
            test_data = test_data.reshape(test_data.shape[0], interprolated_array_len, 12, 1)
            training_data = training_data.reshape(training_data.shape[0], interprolated_array_len, 12, 1)
            input_shape = (interprolated_array_len, 12, 1)

        model = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(2, 2),  activation='relu', input_shape=input_shape),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        verbose = 0
        if debug: verbose = 1
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # TODO: save training accuracy history
        history = model.fit(training_data, self.train_classifications, epochs=50, verbose=verbose)

        if plot_epoch_acc:
            print(history.history['acc'])
            print(history.history.keys())
            #  "Accuracy"
            plt.plot(history.history['acc'])
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['accuracy', 'loss'], loc='upper left')
            plt.show()
            # "Loss"
            # plt.plot(history.history['loss'])
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('epoch')
            # plt.legend(['loss'], loc='upper left')
            # plt.show()

        training_accuracy = history.history['acc']
        # print(training_accuracy)

        # test_loss, test_acc = model.evaluate(test_data, self.test_classifications, verbose=verbose)
        # if debug: print('Test accuracy:', test_acc, "test_loss:",test_loss)

        for _ in self.training_vowels:
            self.training_vowels_accuracy.append([0,0])

        for i in range(len(self.test_data)):
            if not interprolate_coeffs:
                test_data = np.array(self.test_data[i]).reshape(1, coeff_vectors_to_include, 12, 1)
            else:
                test_data = np.array(self.test_data[i]).reshape(1, interprolated_array_len, 12, 1)

            prediction = np.argmax(model.predict(test_data))
            # print("prediction:",prediction, " correct: ",self.test_classifications[i])
            if prediction == self.test_classifications[i]:
                self.training_vowels_accuracy[self.test_classifications[i]][0] += 1
            else:
                self.training_vowels_accuracy[self.test_classifications[i]][1] += 1

        # return test_acc, self.training_vowels_accuracy
        return training_accuracy, self.training_vowels_accuracy

def test_method(trainer, method, vowels, pct_to_train=.75):

    print("\nmethod:", method)
    res = trainer.train_predict(pct_to_train=pct_to_train, method=method)
    for i in range(len(res)):
        vowel = vowels[i]
        corr_cnt = res[i][0]
        err_cnt = res[i][1]
        corr_pct = corr_cnt/float(corr_cnt + err_cnt)
        print("  ",vowel," corr %:",corr_pct)

if __name__ == "__main__":

    # Naming convention
    # sound_name: 'b03ei'
    # file_name: 'vowels/audioclips/m01ae.wav'
    # about ~ 30 length 12 vectors for average mfc_coeffs output
    # all methods: ['LinearSVC',  'LinearSVR',   'NuSVC', 'NuSVR', 'SVC', 'SVR', "LDA"
    # LinearSVC and LinearSVR perform very poorly
    # LDA is not working as expected, only predicts '1'

    # for method in [ 'NuSVC', 'NuSVR', 'SVC', 'SVR']:
    #     trainer = SciKit_Trainer(training_vowels)
    #     test_method(trainer, method, training_vowels)

    # trainer = SciKit_Trainer(training_vowels)
    # trainer.train_predict()

    # --------------------------------------------------------------------------------------------- tensorflow classification

    def print_training_batch_accuracy(training_vowels, n, accuracies ):
        print_str = "   training batch "+str(n)+" with training vowels "+ str(training_vowels) + " [corr,err]: "
        for acc_err in accuracies:
            print_str += str(acc_err[0])+" "+str(acc_err[1])+",\t"
        print(print_str)

    def print_settings(training_vowels, pcd_to_train, normalize, interprolate, num_training_batches):
        print("training vowels:",training_vowels)
        print("training %:",pcd_to_train)
        print("normalize:",normalize)
        print("interprolate:",interprolate)
        print("number of training batches:",num_training_batches)
        print()

    def print_full_batch_accuracy(iterating_val, training_vowel_accuracy, round_=6, batch_time=-1.0):
        print_str = "\nfor iterating val: "+ str(iterating_val) + " ave accuracy: "
        for tv_i in range(training_vowel_accuracy.shape[0]):
            print_str += str( round(training_vowel_accuracy[tv_i, :].mean(),round_) ) +" ± " + str( round(training_vowel_accuracy[tv_i, :].std(),round_) ) +",\t"
        print_str += "    ave batch time:"+str(batch_time)
        print(print_str)
        print_str = "                             "
        for tv_i in range(training_vowel_accuracy.shape[0]):
            print_str += str( round(training_vowel_accuracy[tv_i, :].mean(),round_) ) +"\t" + str( round(training_vowel_accuracy[tv_i, :].std(),round_) ) +"\t"
        print_str += "\n---\n"
        print(print_str)

    # [ "ae", "ah", "aw", "eh", "ei", "er", "ih", "iy", "oa", "oo", "uh", "uw" ]
    training_vowels = ["oa","iy", "oo", "er"]

    round_ = 5
    pct_to_train = .85

    # normalize_mfccs = False
    num_training_batches = 5
    INTERPROLATE = True




    acc, training_vowel_accuracy_i = TF_Trainer(training_vowels).train_predict(pct_to_train=pct_to_train,
                                                                               normalize_mfccs=True,
                                                                               coeff_vectors_to_include=6,
                                                                               debug=True,
                                                                               plot_epoch_acc=True)

    for normalize_mfccs in [True]:

        print("\n >> Starting new hyperparameter sweep\n-------------------------------\n")

        accuracy_stddev_batcht = []

        # Iterate over interprolation lengths
        for iterating_val in range(4, 15):

            training_vowel_accuracy = np.zeros((len(training_vowels), num_training_batches))

            batcht_start = time.time()

            # perform n training_batches for each interprolation length
            for n in range(num_training_batches):

                # get accuracy for each vowel
                if INTERPROLATE:
                    acc, training_vowel_accuracy_i = TF_Trainer(training_vowels).train_predict(pct_to_train=pct_to_train,
                                                                                               interprolate_coeffs=True,
                                                                                               interprolated_array_len=iterating_val,
                                                                                               normalize_mfccs=normalize_mfccs)
                else:
                    acc, training_vowel_accuracy_i = TF_Trainer(training_vowels).train_predict(pct_to_train=pct_to_train,
                                                                                               normalize_mfccs=normalize_mfccs,
                                                                                               coeff_vectors_to_include=iterating_val)

                for tv_i in range(len(training_vowels)):
                    vowel_accuracy = 100*training_vowel_accuracy_i[tv_i][0]/(training_vowel_accuracy_i[tv_i][0]+training_vowel_accuracy_i[tv_i][1])
                    training_vowel_accuracy[tv_i, n] = vowel_accuracy

                print_training_batch_accuracy(training_vowels, n, training_vowel_accuracy_i )

            time_p_batch = (time.time()-batcht_start)/num_training_batches
            print_full_batch_accuracy(iterating_val, training_vowel_accuracy, batch_time=time_p_batch, round_=round_)

            acc_std_t_i = [iterating_val]
            for tv_i in range(training_vowel_accuracy.shape[0]):
                acc_std_t_i.append(training_vowel_accuracy[tv_i, :].mean())
                acc_std_t_i.append(training_vowel_accuracy[tv_i, :].std())
            acc_std_t_i.append(time_p_batch)
            accuracy_stddev_batcht.append(acc_std_t_i)

        print("__________________________________\nResults:")
        print_settings(training_vowels, pct_to_train, normalize_mfccs, INTERPROLATE, num_training_batches)

        for acc_std_t_i in accuracy_stddev_batcht:
            print_str = ""
            for i in acc_std_t_i:
                print_str += str(round(i,round_))+ "\t"
            print(print_str)
        print("__________________________________")



