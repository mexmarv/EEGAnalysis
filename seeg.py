import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, coherence
import edflib  # Using edflib library
from io import BytesIO

# Function to filter the signal
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to calculate band power
def bandpower(data, fs, band):
    f, Pxx = welch(data, fs=fs, nperseg=1024)
    freq_ix = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[freq_ix], f[freq_ix])

# Function to detect peaks in the signal
def detect_peaks(data, height=None, distance=None):
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

# Function to calculate coherence between two signals
def calculate_coherence(signal1, signal2, fs):
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=1024)
    return f, Cxy

# Caching the data
@st.cache_data
def load_edf(file):
    try:
        # Use BytesIO to handle the uploaded file
        with BytesIO(file.read()) as tmpfile:
            with edflib.EdfReader(tmpfile) as f:
                signals = [f.read_signal(i) for i in range(f.num_signals)]
                signal_labels = f.signal_labels
                fs = f.sample_rate
        return signals, signal_labels, fs
    except Exception as e:
        st.error(f"Error al procesar el archivo EDF: {e}")
        return None, None, None

# Streamlit App
st.title("Análisis de EEG.")
st.write("Sube un archivo EDF para analizar las señales EEG.")

uploaded_file = st.file_uploader("Elige un archivo EDF", type=["edf"])

if uploaded_file is not None:
    with st.spinner('Cargando archivo...'):
        signals, signal_labels, fs = load_edf(uploaded_file)

    if signals is not None:
        st.sidebar.title("Seleccionar Canal para Analizar")
        selected_channel = st.sidebar.selectbox("Selecciona un canal:", signal_labels)

        if selected_channel:
            channel_index = signal_labels.index(selected_channel)
            signal = signals[channel_index]

            # Filtrado - banda pasa (0.5-50 Hz)
            filtered_signal = bandpass_filter(signal, 0.5, 50, fs)

            # Asegurarse de que la longitud de la señal filtrada coincida con la original
            filtered_signal = filtered_signal[:len(signal)]

            # Análisis en bandas de frecuencia
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
            band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

            # Calcula la potencia en cada banda
            band_power = {name: bandpower(filtered_signal, fs, band) for name, band in zip(band_names, bands)}

            # Calcular la potencia total
            total_power = sum(band_power.values())

            # Calcular el porcentaje de cada banda
            band_percentage = {name: (power / total_power) * 100 for name, power in band_power.items()}

            # Detectar picos
            peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)

            # Seleccionar los primeros 10 minutos de datos
            duration_in_seconds = 10 * 60  # 10 minutos
            if len(filtered_signal) > duration_in_seconds * fs:
                filtered_signal_10min = filtered_signal[:int(duration_in_seconds * fs)]
            else:
                filtered_signal_10min = filtered_signal

            # Detectar picos en los primeros 10 minutos
            peaks_10min = detect_peaks(filtered_signal_10min, height=np.std(filtered_signal_10min), distance=fs//2)

            # Calcular media de picos por minuto en los primeros 10 minutos
            total_peaks_10min = len(peaks_10min)
            mean_peaks_per_min = total_peaks_10min / 10

            # Mostrar resultados
            st.header(f"Canal: {selected_channel}")

            st.subheader("Porcentaje en Bandas de Frecuencia")
            for band_name, percentage in band_percentage.items():
                st.markdown(f"<span style='color:blue; font-weight:bold;'>{band_name}:</span> {percentage:.2f}%", unsafe_allow_html=True)

            st.subheader("Picos Detectados")
            st.write(f"Cantidad de picos detectados: {len(peaks)}")

            st.subheader("Picos en los Primeros 10 Minutos")
            st.write(f"Total de picos en los primeros 10 minutos: {total_peaks_10min}")
            st.write(f"Media de picos por minuto (Hz) en los primeros 10 minutos: {mean_peaks_per_min:.2f}")

            # Graficar señal filtrada y picos
            fig, ax = plt.subplots()
            ax.plot(signal, label='Señal original')
            ax.plot(filtered_signal, label='Señal filtrada')
            valid_peaks = [p for p in peaks if p < len(filtered_signal)]
            ax.plot(valid_peaks, filtered_signal[valid_peaks], "x", label='Picos detectados')
            ax.set_title(f"Señal y Picos Detectados - {selected_channel}")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Coherencia con otro canal")
            other_channel = st.selectbox("Selecciona otro canal para análisis de coherencia", [ch for ch in signal_labels if ch != selected_channel])

            if other_channel:
                other_channel_index = signal_labels.index(other_channel)
                other_signal = signals[other_channel_index]
                f_coh, Cxy = calculate_coherence(filtered_signal, other_signal, fs)

                fig, ax = plt.subplots()
                ax.plot(f_coh, Cxy)
                ax.set_title(f"Coherencia - {selected_channel} y {other_channel}")
                ax.set_xlabel('Frecuencia (Hz)')
                ax.set_ylabel('Coherencia')
                st.pyplot(fig)

        if st.checkbox("Topographic Option"):
            import mne  # Import mne here since it is optional
            montage = mne.channels.make_standard_montage('standard_1020')
            missing_channels = set(montage.ch_names) - set(signal_labels)

            if missing_channels:
                st.warning(f"No puedo graficar el Topográfico porque los canales faltantes son: {', '.join(missing_channels)}")
            else:
                try:
                    info = mne.create_info(ch_names=signal_labels, sfreq=fs, ch_types='eeg')
                    raw = mne.io.RawArray(np.array(signals).T, info)
                    raw.set_montage(montage)

                    fig = raw.plot_psd(picks='eeg', show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al crear la gráfica topográfica: {e}")
        else:
            st.warning("Activar la opción Topográfico para mostrar la gráfica topográfica.")
    else:
        st.error("No se pudo cargar el archivo EDF.")
