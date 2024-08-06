import streamlit as st
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, coherence
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
import mne

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
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(file.read())
        tmpfile.flush()
        tmpfile.seek(0)
        try:
            f = pyedflib.EdfReader(tmpfile.name)
            signals = [f.readSignal(i) for i in range(f.signals_in_file)]
            signal_labels = f.getSignalLabels()
            fs = f.getSampleFrequency(0)
            f._close()
            return signals, signal_labels, fs
        except Exception as e:
            st.error(f"Error reading EDF file: {e}")
            return None, None, None

# Streamlit App
st.title("Análisis de EEG.")
st.write("Sube un archivo EDF para analizar las señales EEG.")

uploaded_file = st.file_uploader("Elige un archivo EDF", type=["edf"])

if uploaded_file is not None:
    with st.spinner('Cargando archivo...'):
        signals, signal_labels, fs = load_edf(uploaded_file)
        if signals is None:
            st.error("No se pudo leer el archivo EDF.")
        else:
            st.sidebar.title("Seleccionar Canal para Analizar")
            selected_channel = st.sidebar.selectbox("Selecciona un canal", signal_labels)

            selected_index = signal_labels.index(selected_channel)
            signal = signals[selected_index]

            # Plot the raw signal
            st.subheader(f"Señal Raw - {selected_channel}")
            fig, ax = plt.subplots()
            ax.plot(signal)
            ax.set_title(f"Señal Raw - {selected_channel}")
            st.pyplot(fig)

            # Bandpass filter the signal
            lowcut = st.sidebar.slider("Lowcut", 0.1, 30.0, 0.5)
            highcut = st.sidebar.slider("Highcut", 30.0, 100.0, 50.0)
            filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)

            # Plot the filtered signal
            st.subheader(f"Señal Filtrada - {selected_channel}")
            fig, ax = plt.subplots()
            ax.plot(filtered_signal)
            ax.set_title(f"Señal Filtrada - {selected_channel}")
            st.pyplot(fig)

            # Calculate and plot band power
            st.subheader("Potencia de Banda")
            bands = {
                "Delta (0.5-4 Hz)": (0.5, 4),
                "Theta (4-8 Hz)": (4, 8),
                "Alpha (8-12 Hz)": (8, 12),
                "Beta (12-30 Hz)": (12, 30),
                "Gamma (30-100 Hz)": (30, 100),
            }

            band_powers = {band: bandpower(filtered_signal, fs, freq) for band, freq in bands.items()}
            fig, ax = plt.subplots()
            ax.bar(band_powers.keys(), band_powers.values())
            ax.set_title("Potencia de Banda")
            ax.set_ylabel("Potencia")
            st.pyplot(fig)

            # Detect peaks
            st.subheader("Detección de Picos")
            peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)
            fig, ax = plt.subplots()
            ax.plot(filtered_signal)
            ax.plot(peaks, filtered_signal[peaks], "x")
            ax.set_title(f"Picos Detectados - {selected_channel}")
            st.pyplot(fig)

            # Calculate and plot coherence with another channel
            if len(signal_labels) > 1:
                st.subheader("Coherencia con Otro Canal")
                other_channel = st.sidebar.selectbox("Selecciona otro canal", [label for label in signal_labels if label != selected_channel])
                other_index = signal_labels.index(other_channel)
                other_signal = signals[other_index]

                f, Cxy = calculate_coherence(filtered_signal, other_signal, fs)
                fig, ax = plt.subplots()
                ax.plot(f, Cxy)
                ax.set_title(f"Coherencia - {selected_channel} y {other_channel}")
                ax.set_xlabel("Frecuencia (Hz)")
                ax.set_ylabel("Coherencia")
                st.pyplot(fig)

            # Generate PDF report
            st.subheader("Generar Reporte PDF")
            if st.button("Generar Reporte"):
                with st.spinner("Generando reporte..."):
                    pdf_filename = "reporte_eeg.pdf"
                    with PdfPages(pdf_filename) as pdf:
                        for i, signal in enumerate(signals):
                            signal_label = signal_labels[i]
                            filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)

                            fig, ax = plt.subplots()
                            ax.plot(signal, label='Señal original')
                            ax.plot(filtered_signal, label='Señal filtrada')
                            peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)
                            ax.plot(peaks, filtered_signal[peaks], "x", label='Picos detectados')
                            ax.set_title(f"Señal y Picos Detectados - {signal_label}")
                            ax.legend()
                            pdf.savefig(fig)
                            plt.close(fig)

                            if i < len(signal_labels) - 1:
                                other_signal = signals[i + 1]
                                f_coh, Cxy = calculate_coherence(filtered_signal, other_signal, fs)
                                fig, ax = plt.subplots()
                                ax.plot(f_coh, Cxy)
                                ax.set_title(f"Coherencia - {signal_label} y {signal_labels[i + 1]}")
                                ax.set_xlabel('Frecuencia (Hz)')
                                ax.set_ylabel('Coherencia')
                                pdf.savefig(fig)
                                plt.close(fig)

                    st.success("El reporte PDF se ha generado correctamente.")
                    with open(pdf_filename, "rb") as file:
                        btn = st.download_button(
                            label="Descargar Reporte Completo",
                            data=file,
                            file_name=pdf_filename,
                            mime="application/octet-stream"
                        )
