import streamlit as st
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, coherence
import tempfile
from matplotlib.backends.backend_pdf import PdfPages

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
st.title("Análisis de Archivos EDF")
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
        summary = {}

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
            
            # Guardar resumen para los canales específicos
            if signal_labels[ch] in ['C3-P3', 'C4-P4']:
                side = 'Derecho' if signal_labels[ch] == 'C3-P3' else 'Izquierdo'
                summary[signal_labels[ch]] = {
                    'side': side,
                    'total_peaks_10min': total_peaks_10min,
                    'mean_peaks_per_min': mean_peaks_per_min,
                    'alpha_percentage': band_percentage['Alpha']
                }

            # Guarda los resultados del canal
            results.append({
                'name': signal_labels[ch],
                'band_percentage': band_percentage,
                'peaks': peaks,
                'coherence': None  # Inicializar con None
            })

        # Mostrar resumen al principio
        st.header(f"Archivo: {uploaded_file.name}")

        st.subheader("Análisis del Valor de Alpha en Hz")
        for key, value in summary.items():
            st.write(f"**Canal {key} ({value['side']})**")
            st.write(f"Total de picos en los primeros 10 minutos: {value['total_peaks_10min']}")
            st.write(f"Media de picos por minuto (Hz) en los primeros 10 minutos: {value['mean_peaks_per_min']:.2f}")
            st.write(f"Porcentaje de Canal Alpha: {value['alpha_percentage']:.2f}%")

        # Preparar PDF
        pdf_filename = "EEG_analysis.pdf"
        pdf = PdfPages(pdf_filename)

        # Mostrar resultados y gráficos en Streamlit
        for channel in results:
            st.header(f"Canal: {channel['name']}")

            # Mostrar porcentajes de bandas
            st.subheader("Porcentaje en Bandas de Frecuencia")
            for band_name, percentage in channel['band_percentage'].items():
                st.markdown(f"<span style='color:blue; font-weight:bold;'>{band_name}:</span> {percentage:.2f}%", unsafe_allow_html=True)

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
