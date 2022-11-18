# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:39:41 2022

@author: NavigateSafetyField
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def detection(data,sampling_rate,detect_freq,fp,fs,gpass=3,gstop=40):
    """
    Parameters
    ----------
    data : 1D numpy array
        detection data
    sampling_rate : flaot
        detection data sampling rate.
    detect_freq : TYPE
        detection frequency.
    fp : TYPE
        low pass filter parameter.
    fs : TYPE
        low pass filter parameter.
    gpass : TYPE, optional
        low pass filter parameter.
    gstop : TYPE, optional
        low pass filter parameter.

    Returns
    -------
    demod_amp : 1D numpy array
        demodulated data amplitude.
    demod_phase : 1D numpy array
        demodulated data phase. 

    """
    
    # Initialize wave data for Quadrature detection
    time = np.arange(0,data.shape[0]/sampling_rate,1/sampling_rate)
    detect_sin_wav = np.sin(2*np.pi*detect_freq*time)
    detect_cos_wav = np.cos(2*np.pi*detect_freq*time)
    # Multiply wave data
    I = data*detect_sin_wav
    Q = data*detect_cos_wav
    # Low pass filter
    I_filted = __lowpass__(I,sampling_rate,fp,fs,gpass,gstop)
    Q_filted = __lowpass__(Q,sampling_rate,fp,fs,gpass,gstop)
    # Calculate amplitude and init phase    
    demod_amp = np.sqrt(np.power(I_filted*2,2) + np.power(Q_filted*2,2))
    demod_phase = np.arctan2(Q_filted,I_filted)

    return demod_amp,demod_phase

def get_max_init_phase(data,sampling_rate,detect_freq,fp,fs,gpass=3,gstop=40):
    """
    Parameters
    ----------
    data : 1D numpy array
        detection data
    sampling_rate : flaot
        detection data sampling rate.
    detect_freq : TYPE
        detection frequency.
    fp : TYPE
        low pass filter parameter.
    fs : TYPE
        low pass filter parameter.
    gpass : TYPE, optional
        low pass filter parameter.
    gstop : TYPE, optional
        low pass filter parameter.


    Returns
    -------
    init_phase : float
        init phase of max amplitude(-pi < init_phase < pi)

    """
    # Calc quadrature detection
    demod_amp,demod_phase = detection(data,sampling_rate,detect_freq,fp,fs,gpass,gstop)
    # Calc init phase of max amplitude
    init_phase = demod_phase[np.argmax(demod_amp)]
    return init_phase

def get_amp_max_time(data,sampling_rate,detect_freq,fp,fs,gpass=3,gstop=40):
    """
    Parameters
    ----------
    data : 1D numpy array
        detection data
    sampling_rate : flaot
        detection data sampling rate.
    detect_freq : TYPE
        detection frequency.
    fp : TYPE
        low pass filter parameter.
    fs : TYPE
        low pass filter parameter.
    gpass : TYPE, optional
        low pass filter parameter.
    gstop : TYPE, optional
        low pass filter parameter.


    Returns
    -------
    amp_max_time : float
        time of amplitude max.

    """
    # Calc quadrature detection
    demod_amp,_ = detection(data,sampling_rate,detect_freq,fp,fs,gpass,gstop)
    # Calc time of amplitude max time
    amp_max_time = np.argmax(demod_amp)/sampling_rate
    return amp_max_time

def __lowpass__(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2 #ナイキスト周波数
    wp = fp / fn        #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn        #ナイキスト周波数で阻止域端周波数を正規化
    
    N, Wn = signal.buttord(wp, ws, gpass, gstop) #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                 #信号に対してフィルタをかける
    return y                                     #フィルタ後の信号を返す

def main():
    # Make test wave data
    sr = 200*pow(10,3)
    f_mod = 4000
    f_detect = 4000
    send_amp = 2
    noise_cor = 0.1
    wave_time = np.arange(0,0.1,1/sr)
    no_signal_time = np.arange(0,0.1,1/sr)
    init_phase = np.pi/2
    send_wave_time = np.concatenate([no_signal_time,no_signal_time[-1]+wave_time],axis=0)
    
    wave = send_amp*np.sin(2*np.pi*f_mod*wave_time+init_phase)
    window = np.hanning(wave_time.shape[0])
    no_signal_data = np.zeros_like(no_signal_time)
    
    send_signal = np.concatenate([no_signal_data,wave*window],axis=0)
    send_signal = send_signal + noise_cor*np.random.random(send_signal.shape[0])

    # Test functions    
    demod_amp,demod_phase = detection(send_signal,sr,f_detect,1000,2000,gpass=3,gstop=40)
    init_phase = get_max_init_phase(send_signal,sr,f_detect,1000,2000,gpass=3,gstop=40)
    amp_max_time = get_amp_max_time(send_signal,sr,f_detect,1000,2000,gpass=3,gstop=40)

    fig,ax = plt.subplots(4,1,tight_layout=True)
    fig.suptitle("Test Data")
    ax[0].set_title("send wave")
    ax[0].plot(send_wave_time,send_signal,'-')
    ax[1].set_title("detected amp")
    ax[1].plot(send_wave_time,demod_amp,'-')
    ax[2].set_title("detected phase")
    ax[2].plot(send_wave_time,demod_phase,'-')
    ax[2].set_ylim([-np.pi-1,np.pi+1])
    ax[2].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax[3].axis('off')
    ax[3].text(0.5,0.5,"init phase:{phase}\nmax amp time:{amp}".format(phase=init_phase,amp=amp_max_time),
               horizontalalignment="center",verticalalignment="center")
    plt.show()

if __name__ == "__main__":   
    main()
