# Submission 1: Cancer Detection
Nama:Putu Padmanaba

Username dicoding:Putu Padmanaba

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) |
| Masalah | Kanker adalah kondisi patologis yang serius dan dapat mengancam jiwa. Pengembangan sistem prediksi menggunakan machine learning untuk membantu dalam deteksi dini dan prognosis penyakit kanker. |
| Solusi machine learning | Kita dapat memanfaatkan teknologi Machine Learning dalam memecahkan masalah yang ada. Dengan adanya sistem yang bisa memprediksi penyakit cancer sejak dini, dapat membantu dalam pencegahan penyakit lebih awal. |
| Metode pengolahan | Data yang digunakan dalam proyek ini memiliki 1 tipe data, yaitu data numerical. Terdapat 8 fitur yang akan digunakan untuk melatih model. Sebelum data dimasukan ke dalam model, data dinormalisasikan kedalam range yang sama.   |
| Arsitektur model | Pada pipeline ini menggunakan model neural network sederhana yang terdiri dari dense layer, dropout layer, dan memiliki 1 output layer. |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi model adalah AUC, Precision, Recall, ExampleCount, BinaryAccuracy. |
| Performa model | Model yang dihasilkan pipeline memiliki performa yang cukup baik, yaitu mendapatkan BinaryAccuracy sebesar 90.1% |
| Opsi deployment | Proyek machine learning ini dideploy menggunakan salah satu platfrom as a service yaitu KOYEB yang menyediakan layanan gratis untuk mendeploy sebuah proyek. |
| Web app | [cancer-failure-models](https://male-kaleena-naba231-54e6cd2a.koyeb.app/v1/models/cancer-failure-model/metadata)|
| Monitoring | Monitoring proyek ini menggunakan layanan open-source seperti Prometheus, yang memungkinkan pemantauan yang efektif terhadap jumlah request yang masuk ke sistem. Ini memungkinkan untuk melacak perubahan dalam aktivitas sistem dan menampilkan status setiap request yang diterima secara detail. |
