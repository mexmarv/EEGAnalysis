import streamlit as st
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
        f = mne.io.read_raw_edf(tmpfile.name, preload=True)
        signals = f.get_data()
        signal_labels = f.ch_names
        fs = int(f.info['sfreq'])
    return signals, signal_labels, fs

# Streamlit App
st.title("Análisis de EEG.")
st.write("Sube un archivo EDF para analizar las señales EEG.")

uploaded_file = st.file_uploader("Elige un archivo EDF", type=["edf"])

if uploaded_file is not None:
    with st.spinner('Cargando archivo...'):
        try:
            signals, signal_labels, fs = load_edf(uploaded_file)
        except Exception as e:
            st.error(f"Error al cargar el archivo EDF: {e}")
            st.stop()

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
        montage = mne.channels.make_standard_montage('standard_1020')
        missing_channels = set(montage.ch_names) - set(signal_labels)

        if missing_channels:
            st.warning(f"No puedo graficar el Topográfico porque los canales faltantes son: {', '.join(missing_channels)}")
        else:
            try:
                info = mne.create_info(ch_names=signal_labels, sfreq=fs, ch_types='eeg')
                raw = mne.io.RawArray(np.array(signals), info)
                raw.set_montage(montage, on_missing='ignore')

                # Plot power spectral density (PSD) topomap
                fig_psd = raw.plot_psd_topomap(ch_type='eeg', normalize=True, show=False)

                # Alternatively, plot topomap at a specific time point
                # Here, we use an average over a short period as an example
                tmin, tmax = 0, 60  # Time window in seconds
                epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1, preload=True)
                evoked = epochs.average()
                times = [0.1, 0.2, 0.3]  # Times in seconds at which to plot topomaps
                fig_topomap = evoked.plot_topomap(times=times, ch_type='eeg', show=False)

                # Display the plots in Streamlit
                st.pyplot(fig_psd)
                st.pyplot(fig_topomap)
            except RuntimeError as e:
                st.error(f"Error en el montaje: {e}")

    if st.button("Generar Reporte Completo en PDF"):
        pdf_filename = "Reporte_Completo_EEG.pdf"
        with PdfPages(pdf_filename) as pdf:
            for i, signal_label in enumerate(signal_labels):
                signal = signals[i]
                filtered_signal = bandpass_filter(signal, 0.5, 50, fs)
                filtered_signal = filtered_signal[:len(signal)]
                
                band_power = {name: bandpower(filtered_signal, fs, band) for name, band in zip(band_names, bands)}
                total_power = sum(band_power.values())
                band_percentage = {name: (power / total_power) * 100 for name, power in band_power.items()}
                peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)
                
                duration_in_seconds = 10 * 60
                if len(filtered_signal) > duration_in_seconds * fs:
                    filtered_signal_10min = filtered_signal[:int(duration_in_seconds * fs)]
                else:
                    filtered_signal_10min = filtered_signal
                    
                peaks_10min = detect_peaks(filtered_signal_10min, height=np.std(filtered_signal_10min), distance=fs//2)
                total_peaks_10min = len(peaks_10min)
                mean_peaks_per_min = total_peaks_10min / 10

                fig, ax = plt.subplots()
                ax.plot(signal, label='Señal original')
                ax.plot(filtered_signal, label='Señal filtrada')
                valid_peaks = [p for p in peaks if p < len(filtered_signal)]
                ax.plot(valid_peaks, filtered_signal[valid_peaks], "x", label='Picos detectados')
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
