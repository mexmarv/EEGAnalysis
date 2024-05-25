import streamlit as st
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, coherence
import tempfile
from matplotlib.backends.backend_pdf import PdfPages
import base64

# Función para filtrar la señal
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Función para calcular la potencia de la banda
def bandpower(data, fs, band):
    f, Pxx = welch(data, fs=fs, nperseg=1024)
    freq_ix = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[freq_ix], f[freq_ix])

# Función para detectar picos en la señal
def detect_peaks(data, height=None, distance=None):
    peaks, _ = find_peaks(data, height=height, distance=distance)
    return peaks

# Función para calcular la coherencia entre dos señales
def calculate_coherence(signal1, signal2, fs):
    f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=1024)
    return f, Cxy

# Aplicación Streamlit
import streamlit as st
st.image('https://neuro-praxis-dus.de/media/pages/diagnostik/elektroenzephalographie/modules/eeg-text/5db7120a26-1673453655/emptyname-326-870x-q75.jpg', caption='EEG')
st.title("Análisis Quantitativo de Electroencefalogramas.")
st.write("Por Marvin Nahmias. Bandas, Picos, Filtrado de ruido y Coherencia.")
st.write("Sube un archivo EDF para analizar las señales EEG.")

uploaded_file = st.file_uploader("Elige un archivo EDF", type=["edf"])

if uploaded_file is not None:
    with st.spinner('Cargando archivo...'):
        # Usar un archivo temporal para manejar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile.flush()
            tmpfile.seek(0)

            # Leer el archivo EDF
            f = pyedflib.EdfReader(tmpfile.name)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            fs = f.getSampleFrequency(0)
            signals = [f.readSignal(i) for i in range(n)]
            f._close()

        results = []

        for ch in range(n):
            signal = signals[ch]
            
            # Filtrado - banda pasa (0.5-50 Hz)
            filtered_signal = bandpass_filter(signal, 0.5, 50, fs)
            
            # Asegurarse de que la longitud de la señal filtrada coincida con la original
            filtered_signal = filtered_signal[:len(signal)]
            
            # Análisis en bandas de frecuencia
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
            band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            
            # Calcula la potencia en cada banda
            band_power = {name: bandpower(filtered_signal, fs, band) for name, band in zip(band_names, bands)}
            
            # Detectar picos
            peaks = detect_peaks(filtered_signal, height=np.std(filtered_signal), distance=fs//2)
            
            # Coherencia (ejemplo entre primer y segundo canal si existen)
            if ch < n - 1:
                f_coh, Cxy = calculate_coherence(filtered_signal, signals[ch + 1], fs)
            else:
                f_coh, Cxy = None, None
            
            # Guarda los resultados del canal
            results.append({
                'name': signal_labels[ch],
                'band_power': band_power,
                'peaks': peaks,
                'coherence': (f_coh, Cxy) if f_coh is not None else None
            })

        st.header(f"Archivo: {uploaded_file.name}")
        
        pdf_filename = "EEG_analysis.pdf"
        pdf = PdfPages(pdf_filename)

        # Mostrar resultados y gráficos en Streamlit
        for channel in results:
            st.header(f"Canal: {channel['name']}")
            
            # Mostrar potencias de bandas
            st.subheader("Potencia en Bandas de Frecuencia")
            for band_name, power in channel['band_power'].items():
                st.markdown(f"<span style='color:blue; font-weight:bold;'>{band_name}:</span> {power}", unsafe_allow_html=True)
            
            # Mostrar número de picos detectados
            st.subheader("Picos Detectados")
            st.write(f"Cantidad de picos detectados: {len(channel['peaks'])}")
            
            # Graficar señal filtrada y picos
            fig, ax = plt.subplots()
            ax.plot(signal, label='Señal original')
            ax.plot(filtered_signal, label='Señal filtrada')
            valid_peaks = [p for p in channel['peaks'] if p < len(filtered_signal)]
            ax.plot(valid_peaks, filtered_signal[valid_peaks], "x", label='Picos detectados')
            ax.set_title(f"Señal y Picos Detectados - {channel['name']}")
            ax.legend()
            st.pyplot(fig)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Graficar coherencia si está disponible
            if channel['coherence'] is not None:
                f_coh, Cxy = channel['coherence']
                fig, ax = plt.subplots()
                ax.plot(f_coh, Cxy)
                ax.set_title(f"Coherencia - {channel['name']} y Siguiente Canal")
                ax.set_xlabel('Frecuencia (Hz)')
                ax.set_ylabel('Coherencia')
                st.pyplot(fig)
                pdf.savefig(fig)
                plt.close(fig)

        pdf.close()

        with open(pdf_filename, "rb") as file:
            btn = st.download_button(
                label="Descargar PDF",
                data=file,
                file_name=pdf_filename,
                mime="application/octet-stream"
            )
