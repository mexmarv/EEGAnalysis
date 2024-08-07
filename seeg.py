import streamlit as st
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, coherence
from scipy.interpolate import interp1d
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
import mne

# Caching the data loader function
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_edf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(file.read())
        tmpfile.flush()
        tmpfile.seek(0)
        try:
            f = pyedflib.EdfReader(tmpfile.name)
            signals = []
            for i in range(f.signals_in_file):
                signal = f.readSignal(i)
                if np.any(np.isnan(signal)):
                    # Handle NaNs by interpolation
                    nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
                    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
                signals.append(signal)
            signal_labels = f.getSignalLabels()
            fs = f.getSampleFrequency(0)
            f._close()
            return signals, signal_labels, fs
        except Exception as e:
            st.error(f"Error reading EDF file: {e}")
            return None, None, None

# Caching the signal processing functions
@st.cache
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

@st.cache
def bandpower(data, fs, band):
    f, Pxx = welch(data, fs=fs, nperseg=1024)
    freq_ix = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[freq_ix], f[freq_ix])

@st.cache
def detect_peaks(data, height=None, distance=None):
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

@st.cache
def calculate_coherence(signal1, signal2, fs):
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=1024)
    return f, Cxy

# Streamlit App
st.title("Análisis de EEG.")
st.write("Sube un archivo EDF para analizar las señales EEG.")
st.markdown('**Autores:** Dr. Marina Cardoso y M.Sc. Marvin Nahmias')

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
            peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)

            # Select the first 10 minutes of data
            duration_in_seconds = 10 * 60  # 10 minutes
            if len(filtered_signal) > duration_in_seconds * fs:
                filtered_signal_10min = filtered_signal[:int(duration_in_seconds * fs)]
            else:
                filtered_signal_10min = filtered_signal

            # Detect peaks in the first 10 minutes
            peaks_10min = detect_peaks(filtered_signal_10min, height=np.std(filtered_signal_10min), distance=fs//2)

            # Calculate mean peaks per minute in the first 10 minutes
            total_peaks_10min = len(peaks_10min)
            mean_peaks_per_min = total_peaks_10min / 10

            st.subheader("Detección de Picos en los primeros 10 minutos")
            st.write(f"Cantidad total de picos en los primeros 10 minutos: {total_peaks_10min}")
            st.write(f"Promedio de picos por minuto en los primeros 10 minutos: {mean_peaks_per_min:.2f}")

            fig, ax = plt.subplots()
            ax.plot(filtered_signal_10min)
            ax.plot(peaks_10min, filtered_signal_10min[peaks_10min], "x")
            ax.set_title(f"Picos Detectados en los primeros 10 minutos - {selected_channel}")
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

                            # Plot raw signal
                            fig, ax = plt.subplots()
                            ax.plot(signal)
                            ax.set_title(f"Señal Raw - {signal_label}")
                            pdf.savefig(fig)
                            plt.close(fig)

                            # Plot filtered signal
                            fig, ax = plt.subplots()
                            ax.plot(filtered_signal)
                            ax.set_title(f"Señal Filtrada - {signal_label}")
                            pdf.savefig(fig)
                            plt.close(fig)

                            # Band power
                            band_powers = {band: bandpower(filtered_signal, fs, freq) for band, freq in bands.items()}
                            fig, ax = plt.subplots()
                            ax.bar(band_powers.keys(), band_powers.values())
                            ax.set_title("Potencia de Banda")
                            ax.set_ylabel("Potencia")
                            pdf.savefig(fig)
                            plt.close(fig)

                            # Peaks in the first 10 minutes
                            duration_in_seconds = 10 * 60  # 10 minutes
                            if len(filtered_signal) > duration_in_seconds * fs:
                                filtered_signal_10min = filtered_signal[:int(duration_in_seconds * fs)]
                            else:
                                filtered_signal_10min = filtered_signal
                            peaks_10min = detect_peaks(filtered_signal_10min, height=np.std(filtered_signal_10min), distance=fs//2)
                            total_peaks_10min = len(peaks_10min)
                            mean_peaks_per_min = total_peaks_10min / 10

                            fig, ax = plt.subplots()
                            ax.plot(filtered_signal_10min)
                            ax.plot(peaks_10min, filtered_signal_10min[peaks_10min], "x")
                            ax.set_title(f"Picos Detectados en los primeros 10 minutos - {signal_label}")
                            pdf.savefig(fig)
                            plt.close(fig)

                            # Write band powers and peak analysis to PDF
                            fig = plt.figure(figsize=(8, 6))
                            plt.axis('off')
                            plt.text(0.1, 0.9, f"Potencia de Banda para {signal_label}:", fontsize=12, ha='left')
                            y_text = 0.85
                            for band, power in band_powers.items():
                                plt.text(0.1, y_text, f"{band}: {power:.2f}", fontsize=12, ha='left')
                                y_text -= 0.05

                            plt.text(0.1, 0.5, f"Cantidad total de picos en los primeros 10 minutos: {total_peaks_10min}", fontsize=12, ha='left')
                            plt.text(0.1, 0.45, f"Promedio de picos por minuto en los primeros 10 minutos: {mean_peaks_per_min:.2f}", fontsize=12, ha='left')
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
