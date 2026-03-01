import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="DASBOARD ANALISIS HASIL BELAJAR SISWA",
    layout="wide"
)

st.title("🎓 DASBOARD ANALISIS HASIL BELAJAR SISWA")
st.markdown("Analisis Data 50 Siswa × 20 Soal Secara Interaktif dan Visual")

# ======================================================
# LOAD FILE OTOMATIS (TANPA UPLOAD)
# ======================================================
file_path = "data_simulasi_50_siswa_20_soal.xlsx"

try:
    df = pd.read_excel(file_path)
except:
    st.error("File Excel tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
    st.stop()

# Ambil hanya kolom numerik (20 soal)
df_numeric = df.select_dtypes(include=np.number)

if df_numeric.shape[1] < 20:
    st.error("File harus memiliki minimal 20 kolom soal numerik.")
    st.stop()

data = df_numeric.iloc[:, :20].copy()

# ======================================================
# 1️⃣ SKOR TOTAL
# ======================================================
data["Total_Skor"] = data.sum(axis=1)

st.header("1️⃣ Statistik Nilai Siswa")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Siswa", len(data))
col2.metric("Rata-rata", round(data["Total_Skor"].mean(), 2))
col3.metric("Nilai Tertinggi", data["Total_Skor"].max())
col4.metric("Nilai Terendah", data["Total_Skor"].min())

# Histogram
fig1, ax1 = plt.subplots()
ax1.hist(data["Total_Skor"], bins=10)
ax1.set_title("Distribusi Skor Total")
ax1.set_xlabel("Skor")
ax1.set_ylabel("Frekuensi")
st.pyplot(fig1)

# ======================================================
# 2️⃣ TINGKAT KESUKARAN
# ======================================================
st.header("2️⃣ Analisis Tingkat Kesukaran Soal")

tingkat_kesukaran = data.iloc[:, :20].mean()

fig2, ax2 = plt.subplots(figsize=(10, 4))
tingkat_kesukaran.plot(kind="bar", ax=ax2)
ax2.axhline(tingkat_kesukaran.mean(), linestyle="--")
ax2.set_title("Rata-rata Skor per Soal")
st.pyplot(fig2)

soal_mudah = tingkat_kesukaran.idxmax()
soal_sulit = tingkat_kesukaran.idxmin()

col5, col6 = st.columns(2)
col5.success(f"Soal Paling Mudah: {soal_mudah}")
col6.error(f"Soal Paling Sulit: {soal_sulit}")

# ======================================================
# 3️⃣ DAYA PEMBEDA
# ======================================================
st.header("3️⃣ Analisis Daya Pembeda (Korelasi Item-Total)")

daya_pembeda = {}
for col in data.columns[:20]:
    daya_pembeda[col] = data[col].corr(data["Total_Skor"])

daya_pembeda = pd.Series(daya_pembeda)

fig3, ax3 = plt.subplots(figsize=(10, 4))
daya_pembeda.plot(kind="bar", ax=ax3)
ax3.axhline(0.3, linestyle="--")
ax3.set_title("Daya Pembeda Tiap Soal")
st.pyplot(fig3)

# ======================================================
# 4️⃣ HEATMAP KORELASI
# ======================================================
st.header("4️⃣ Korelasi Antar Soal")

corr = data.iloc[:, :20].corr()

fig4, ax4 = plt.subplots(figsize=(8, 6))
cax = ax4.imshow(corr)
plt.colorbar(cax)
ax4.set_title("Heatmap Korelasi Soal")
st.pyplot(fig4)

# ======================================================
# 5️⃣ CLUSTERING SISWA
# ======================================================
st.header("5️⃣ Segmentasi Kemampuan Siswa")

scaler = StandardScaler()
scaled = scaler.fit_transform(data.iloc[:, :20])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster = kmeans.fit_predict(scaled)

data["Cluster"] = cluster

cluster_mean = data.groupby("Cluster")["Total_Skor"].mean()

fig5, ax5 = plt.subplots()
cluster_mean.plot(kind="bar", ax=ax5)
ax5.set_title("Rata-rata Skor per Cluster")
st.pyplot(fig5)

# ======================================================
# 6️⃣ RANKING SOAL
# ======================================================
st.header("6️⃣ Ranking Soal Berdasarkan Daya Pembeda")

ranking = daya_pembeda.sort_values(ascending=False)
st.dataframe(ranking)

st.success("Analisis Selesai ✅ Dashboard Siap Digunakan")