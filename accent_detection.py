import streamlit as st
import librosa
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk mengekstrak MFCC dari file audio
def extract_mfcc(audio_data, n_mfcc=13):
    # Menyimpan audio sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_data.getbuffer())  # Menulis byte stream ke file
        tmp_file_path = tmp_file.name

    # Memuat audio dan mengekstrak MFCC
    audio, sr = librosa.load(tmp_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # Rata-rata MFCC untuk koefisien
    return mfcc_mean, tmp_file_path  # Mengembalikan path file

# Memuat model yang sudah dilatih
model = load_model('cnn_tunning.h5')  # Gantilah dengan path model Anda

# Memuat LabelEncoder yang digunakan selama pelatihan
label_encoder = LabelEncoder()
label_encoder.fit([
    "arabic", "dutch", "english", "french", "german", 
    "italian", "korean", "mandarin", "polish", "portuguese", 
    "russian", "spanish", "turkish"
])

# Judul aplikasi
st.title("Speech Accent Detection App")
st.markdown("""
    This application allows you to record your voice and predict the accent using a pre-trained model. 
    Simply click the 'Record your voice' button, speak, and the app will show the predicted accent!
    """)

# Perekaman suara dari mikrofon pengguna
audio_value = st.audio_input("Record your voice")

# Handle the case when no audio is recorded yet
if audio_value is None:
    st.info("Please record your voice to get the prediction!")
else:
    # Ekstraksi MFCC dari audio yang direkam dan mendapatkan path file
    mfcc_features, file_path = extract_mfcc(audio_value)
    
    # Reshape data agar sesuai dengan input model CNN
    mfcc_features_reshaped = mfcc_features.reshape(1, mfcc_features.shape[0], 1)

    # Prediksi aksen menggunakan model
    pred_prob = model.predict(mfcc_features_reshaped)
    pred_class = np.argmax(pred_prob, axis=1)

    # Mengonversi hasil prediksi kembali ke label asli
    pred_label = label_encoder.inverse_transform(pred_class)

    # Tampilkan hasil prediksi
    st.success(f"Predicted Accent: **{pred_label[0]}**")



# import streamlit as st
# import librosa
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder

# # Fungsi untuk mengekstrak MFCC dari file audio
# def extract_mfcc(file_path, n_mfcc=13):
#     audio, sr = librosa.load(file_path, sr=None)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     mfcc_mean = np.mean(mfcc, axis=1)  # Rata-rata untuk setiap koefisien MFCC
#     return mfcc_mean

# # Memuat model yang sudah dilatih (gantilah dengan path model Anda)
# model = load_model('cnn.h5')  # Gantilah dengan path model Anda

# # Memuat LabelEncoder yang digunakan selama pelatihan
# label_encoder = LabelEncoder()
# label_encoder.fit([
#     "arabic", "dutch", "english", "french", "german", 
#     "italian", "korean", "mandarin", "polish", "portuguese", 
#     "russian", "spanish", "turkish"
# ])

# # Judul aplikasi
# st.title("Speech Accent Prediction App")
# st.write("Upload an audio file to predict its accent!")

# # Upload file audio
# uploaded_file = st.file_uploader("Choose an audio file (wav)", type=["mp3", "wav"])

# if uploaded_file is not None:
#     # Simpan file audio sementara di server
#     with open("temp_audio.mp3", "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Ekstraksi MFCC dari file audio yang diunggah
#     mfcc_features = extract_mfcc("temp_audio.mp3")
    
#     # Reshape data agar sesuai dengan input model CNN
#     mfcc_features_reshaped = mfcc_features.reshape(1, mfcc_features.shape[0], 1)

#     # Prediksi aksen menggunakan model
#     pred_prob = model.predict(mfcc_features_reshaped)
#     pred_class = np.argmax(pred_prob, axis=1)

#     # Mengonversi hasil prediksi kembali ke label asli
#     pred_label = label_encoder.inverse_transform(pred_class)

#     # Tampilkan hasil prediksi
#     st.write(f"Predicted Accent: {pred_label[0]}")