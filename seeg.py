alpha_percentage': band_percentage['Alpha']
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
