%% sEMG Signal Processing
%% MATLAB Script used to learn to perform FFT on data

clear all;
clc;

%% Import raw data from text file
raw_data = dlmread('M3.txt', '\t', 9, 0);

% Preallocation of resources 
data = zeros(length(raw_data), 3);
time = zeros(length(raw_data), 1);

%%  sEMG time data
time = raw_data(:, 1) ./ 1000;

%% Sensor transfer function - Analog values from digital data
L = length(data);

% Transfer function:
% EMG = (ADC / 2^n) - 0.5) * Vcc/Gain --> n=16 bits, Gain = 1000, Vcc = 3V

for i = 1:L
    data(i, 1) = ((((raw_data(1, 3) / 2^16) - 0.5) * 3) / 1000) * 1000; % Location on arm
    data(i, 2) = ((((raw_data(1, 4) / 2^16) - 0.5) * 3) / 1000) * 1000; % Location on arm
    data(i, 3) = ((((raw_data(1, 5) / 2^16) - 0.5) * 3) / 1000) * 1000; % Location on arm
end

%%  Plotting raw sEMG data

figure;

for i = 1:3
    subplot(3, 1, i);
    plot(time, data(:, i));
    xlabel('Time (s)');
    ylabel('Voltage (mV)');

    grid
    hold on

    if (i == 1)
        title('Lateral Antebrachial')
    elseif (i == 2)
        title('Medial Antebrachial')
    else
        title('')
    end
end

sgtitle('Raw sEMG');
hold off

%% FFT - to find cutoff frequencies for BP filter

fs = 1000; % sampling frequency

% Frequency resolution = fs/L
% Frequency range for FFT
% FFT frequency range defined as frequency resolution from 0 to 1/2 data
% length
f = fs * (0:(L / 2)) / L;

% Finding and plotting FFT

figure;
for i = 1:3
    % Compute 2 sided spectrum (-Fmax:Fmax)
    p1 = fft(data(:, i));

    % Divide by L for mormalization of the power of the output for the
    % length of the input signal
    p1 = abs(p1 / L);
 
    % Compute single sided spectrum by taking the positive part of the
    % double sided spectrum and multiply by 2
    p1 = p1(1:L/2+1);
    p1(2:end-1) = 2*p1(2:end-1);

    subplot(3, 1, i)
    plot(f, p1);
    xlabel('Frequency (Hz)')
    ylabel('Intensity');
    title('')
    grid

    if (i == 1)
        title('Lateral Antebrachial')
    elseif (i == 2)
        title('Medial Antebrachial')
    else
        title('')
    end

    hold on
end

%% BP Filter
% Filtering from 30 to 300 Hz

fnyq = fs/2; % Nyquist frequency
fcuthigh = 30; % Highpass cutoff frequency in Hz (set by looking at plot)
fcutlow = 300; % Lowpass cutoff frequency in Hz (set by looking at plot)

% 4th order butterworth BP filter
[b, a] = butter(4, [fcuthigh, fcutlow] / fnyq, 'bandpass')

for i = 1:3
    data(:, 1) = filtfilt(b, a, data(:, i)); % Apply BP filter to data
end

%% Full wave rectification
% Preallocate resources
rec_signal = zeros(length(data), 3);

for i = 1:3
    rec_signal(:, 1) = abs(data(:, 1));
end

%% Plotting signal after BP filter and rectification

figure;

for i = 1:3
    subplot(3, 1, i);
    plot(time, data(:, i));
    xlabel('Time (s)');
    ylabel('Voltage (mV)');

    grid
    hold on

    if (i == 1)
        title('Lateral Antebrachial')
    elseif (i == 2)
        title('Medial Antebrachial')
    else
        title('')
    end
end

sgtitle('BP Filter and Rectification');
hold off

%% RMS Envolope (moving average) - mean power of signal

envelope = zeros(L, 3); % L = length(data)

% Window size in ms
window = 50; % Larger window size has more of the original signal

for i = 1:3
    envelope(:, i) = sqrt(movmean((rec_signal(:, i).^2), window));
end

%% Normalization (Comparing with MVC)
% MVC in mV
% MVC = [Muscle1, Muscle2, Muscle3]
MVC = [1.5, 1.5, 1.5] ; % Fill with max voluntary contraction values

MVC_normalized = zeros(L, 3);

% Divide signal by MVC and convert to percentage
for i = 1:3
    MVC_normalized(:, i) = envelope(:, i) ./ MVC(1, i) .* 100;
end

%% Plotting normalized envelope against Moving RMS

% Limits (sec) for cropped section of trial (randomly chosen)
lim1 = 5;
lim2 = 10;

figure;

for i = 1:3
    subplot(3, 1, i)
    plot(time, rec_signal(:, 1));
    hold on

    plot(time, envelope(:, i), 'r', 'LineWidth', 2);

    % Cropping Trial
    xlim(lim1, lim2);

    xlabel('Time (s)');
    ylabel('Voltage (mV)')
    grid
    

