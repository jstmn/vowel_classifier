import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import csv
import os
from sklearn import linear_model
from sklearn import svm
import random
import time

class MFCC_Generator(object):


    def __init__(self, sound_file, start_end_t):

        self.sound_file = sound_file
        self.start_t = start_end_t[0]
        self.end_t = start_end_t[1]

    def calculate_mfc_coeffs(self, normalize=True):

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

        sample_rate, signal = scipy.io.wavfile.read(self.sound_file)
        sample_rate = float(sample_rate)

        signal = signal[ int( self.start_t * sample_rate) : int(self.end_t * sample_rate) ]  # Keep the first 3.5 seconds

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
            mfcc / np.max(np.abs(mfcc))
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


class Trainer(object):

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

    vowels = [ "ae", "ah", "aw", "eh", "ei", "er", "ih", "iy", "oa", "oo", "uh", "uw" ]

    def __init__(self, training_vowels):

        self.mcf_coeffs = []
        self.classifier = []

        self.training_vowels = training_vowels
        self.training_vowels_accuracy = [] # [[vowel 1 correct cnt, vowel 1 err count],[vowel2 correct cnt, vowel2 err count],..]

        self.csv_reader = CSVReader(Trainer.time_data_csv_file)

    def get_vowels(self):
        return self.training_vowels

    def train(self, pct_to_train=.9, method="SVC"):

        directory = "vowels/audioclips"

        # only add if # > min_num_to_test
        rand_int_min = 0

        for sound_name in os.listdir(directory):

            sound_name_no_wav = sound_name.split('.')[0]
            file_name = directory+"/"+sound_name

            for vowel in self.training_vowels:

                if vowel in sound_name_no_wav and int(sound_name_no_wav[1:3]) > (1.0-pct_to_train)*50:

                    start_end_t = self.csv_reader.get_start_end_time_from_fname(file_name)
                    mfcc_generator = MFCC_Generator(file_name, start_end_t)
                    ret = mfcc_generator.calculate_mfc_coeffs()
                    for mcf_coeff in ret:
                        if random.randint(0,1000) > rand_int_min:
                            self.mcf_coeffs.append(mcf_coeff)
                            classifier_num = self.training_vowels.index(vowel)
                            self.classifier.append(classifier_num)

        if method == "SVC":
            clf = svm.SVC(gamma='scale')
        elif method == "SVR":
            clf = svm.SVC(gamma='scale')
        elif method == 'LinearSVC':
            clf = svm.LinearSVC()
        elif method == 'LinearSVR':
            clf = svm.LinearSVR()
        elif method == 'NuSVC':
            clf = svm.NuSVC(gamma='scale')
        elif method == 'NuSVR':
            clf = svm.NuSVR(gamma='scale')
        elif method == 'OneClassSVM':
            clf = svm.OneClassSVM(gamma='scale')
        else:
            return

        clf.fit(self.mcf_coeffs, self.classifier)

        for i in self.training_vowels:
            self.training_vowels_accuracy.append([0,0])

        for sound_name in os.listdir(directory):

            sound_name_no_wav = sound_name.split('.')[0]
            file_name = directory+"/"+sound_name

            for vowel in self.training_vowels:

                if vowel in sound_name_no_wav and int(sound_name_no_wav[1:3]) < (1.0-pct_to_train)*50:

                    start_end_t = self.csv_reader.get_start_end_time_from_fname(file_name)
                    mfcc_generator = MFCC_Generator(file_name, start_end_t)
                    ret = mfcc_generator.calculate_mfc_coeffs()
                    classifier_num = self.training_vowels.index(vowel)
                    for coeff in ret:

                        prediction = clf.predict([coeff])
                        if prediction[0] == classifier_num:
                            self.training_vowels_accuracy[classifier_num][0] += 1
                        else:
                            self.training_vowels_accuracy[classifier_num][1] += 1

        return self.training_vowels_accuracy

def test_method(trainer, method, vowels, pct_to_train=.75):

    print "method:", method
    res = trainer.train(pct_to_train=pct_to_train, method=method)
    for i in range(len(res)):
        vowel = vowels[i]
        corr_cnt = res[i][0]
        err_cnt = res[i][1]
        corr_pct = corr_cnt/float(corr_cnt + err_cnt)
        print "  ",vowel," corr %:",corr_pct

if __name__ == "__main__":

    # Naming convention
    # sound_name: 'b03ei'
    # file_name: 'vowels/audioclips/m01ae.wav'
    # about ~ 30 length 12 vectors for average mfc_coeffs output

    # -- Sample code
    # for method in ['LinearSVC',  'LinearSVR',   'NuSVC', 'NuSVR', 'OneClassSVM', 'SVC', 'SVR']:
    #     trainer = Trainer()
    #     test_method(trainer, method)


    training_vowels = ["oo", "er", "ah"]
    trainer = Trainer(training_vowels)
    test_method(trainer,"SVR", training_vowels)





