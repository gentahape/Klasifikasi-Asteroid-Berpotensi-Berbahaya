# Laporan Proyek Machine Learning - Genta Haetami Putra

## Klasifikasi Asteroid Berpotensi Berbahaya (PHA)

Sistem tata surya kita merupakan lingkungan yang dinamis, diisi oleh jutaan benda langit, termasuk asteroid yang sebagian besar tidak berbahaya. Namun, sebagian kecil dari asteroid ini, yang dikenal sebagai Objek Dekat Bumi (Near-Earth Objects atau NEO), memiliki orbit yang memotong atau mendekati orbit Bumi. Dari kelompok ini, objek yang berukuran cukup besar dan memiliki jarak orbit minimum yang sangat dekat dengan Bumi diklasifikasikan sebagai Asteroid Berpotensi Berbahaya (Potentially Hazardous Asteroids atau PHA). Mengapa masalah ini penting untuk diselesaikan? Karena sejarah geologis Bumi dan peristiwa modern seperti meteor Chelyabinsk pada tahun 2013 menjadi pengingat nyata bahwa dampak dari objek semacam itu, meskipun jarang, dapat menyebabkan kerusakan berskala regional hingga global. Oleh karena itu, identifikasi dan pemantauan PHA bukan hanya sekadar rasa ingin tahu astronomis, melainkan komponen penting dari upaya pertahanan planet (planetary defense) untuk mitigasi bencana di masa depan.

Menanggapi potensi ancaman ini, badan antariksa global seperti NASA telah menginisiasi program untuk menemukan, melacak, dan mengkarakterisasi populasi NEO. Kemajuan teknologi dalam survei langit telah menghasilkan peningkatan eksponensial dalam laju penemuan asteroid baru, yang menciptakan volume data yang sangat besar. Tantangan yang muncul adalah bagaimana cara menyelesaikan masalah ini? Proses analisis manual untuk setiap objek baru tidak lagi efisien untuk menangani skala dan kecepatan data yang masuk. Diperlukan sebuah sistem otomatis yang dapat dengan cepat menganalisis data awal setiap asteroid yang baru ditemukan untuk memberikan penilaian risiko awal, sehingga para astronom dapat memfokuskan sumber daya pengamatan yang terbatas pada objek-objek yang paling berisiko.

Solusi modern untuk tantangan ini adalah dengan menerapkan pendekatan machine learning. Dengan memanfaatkan data historis dari asteroid yang telah diketahui, kita dapat melatih sebuah model klasifikasi untuk mengenali pola kompleks yang membedakan antara asteroid berbahaya dan yang tidak berbahaya. Studi-studi terkini, seperti yang dilakukan oleh Erasmus dkk. [1], telah menunjukkan bahwa algoritma machine learning mampu mengklasifikasikan asteroid dekat Bumi dengan tingkat akurasi yang tinggi berdasarkan parameter fotometrik dan orbitalnya. Implementasi sistem klasifikasi otomatis ini akan menjadi alat bantu yang sangat berharga untuk menyaring data secara efisien, memberikan peringatan dini yang lebih cepat, dan meningkatkan efektivitas program pertahanan planet secara keseluruhan.

Referensi
[1] N. Erasmus, S. McNeill, M. Mommert, and D. Trilling, "Machine learning classification of near-Earth asteroids," arXiv preprint arXiv:1804.05389, 2018.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan, terdapat dua masalah utama yang perlu diselesaikan:
- Pernyataan Masalah 1: Peningkatan volume data dari survei langit modern telah menyebabkan penemuan ribuan asteroid baru secara terus-menerus. Proses analisis manual untuk mengevaluasi tingkat ancaman setiap asteroid menjadi tidak efisien, memakan waktu, dan sulit untuk diskalakan, sehingga berisiko melewatkan objek berbahaya.
- Pernyataan Masalah 2: Dibutuhkan sebuah metode yang konsisten dan objektif untuk melakukan penilaian risiko awal. Penilaian yang bergantung sepenuhnya pada pengamat manusia dapat bersifat subjektif dan tidak dapat memberikan respons yang cukup cepat ketika sebuah objek berpotensi berbahaya terdeteksi.

### Goals
Untuk menjawab pernyataan masalah tersebut, tujuan dari proyek ini adalah sebagai berikut:
- Tujuan untuk Pernyataan Masalah 1: Mengembangkan sebuah model machine learning yang mampu mengolah dan mempelajari pola dari dataset besar berisi karakteristik fisis dan orbital asteroid, guna membedakan antara asteroid yang berpotensi berbahaya dan yang tidak.
- Tujuan untuk Pernyataan Masalah 2: Membangun sebuah sistem klasifikasi otomatis yang dapat diandalkan untuk menetapkan label risiko ("Berbahaya" atau "Tidak Berbahaya") pada asteroid. Sistem ini diharapkan dapat menjadi alat bantu skrining yang cepat dan efisien bagi para astronom.

### Solution Statements
Untuk mencapai tujuan tersebut, diajukan dua pendekatan solusi (solution statements) yang akan dikembangkan dan dievaluasi kinerjanya:
- Solusi 1: Membuat Model Baseline dengan Logistic Regression
Membangun model klasifikasi dasar menggunakan algoritma Logistic Regression. Model ini dipilih karena sederhana, cepat untuk dilatih, dan hasilnya mudah diinterpretasikan. Model ini akan menjadi titik acuan (baseline) untuk mengukur kinerja model yang lebih kompleks.
- Solusi 2: Membuat Model Ensemble dengan Random Forest Classifier
Membangun model klasifikasi yang lebih canggih menggunakan algoritma ensemble Random Forest Classifier. Model ini dipilih karena kemampuannya menangani hubungan non-linear yang kompleks dalam data dan umumnya memiliki performa yang lebih tinggi.

Kinerja kedua model akan dievaluasi dan dibandingkan berdasarkan metrik yang sesuai untuk masalah klasifikasi, yaitu Accuracy, Precision, Recall, dan F1-Score. Model dengan performa terbaik berdasarkan metrik-metrik ini akan dipilih sebagai solusi akhir untuk masalah klasifikasi Asteroid Berpotensi Berbahaya (PHA).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Possible Asteroid Impacts with Earth" yang disediakan oleh NASA. Data ini berisi informasi mengenai karakteristik fisis dan parameter orbit dari asteroid-asteroid yang memiliki kemungkinan berdampak dengan Bumi, yang dipantau oleh sistem Sentry JPL (Jet Propulsion Laboratory). Dataset ini dapat diakses dan diunduh secara publik melalui platform Kaggle.
Sumber Data: Kaggle: [NASA Asteroid Impacts](https://www.kaggle.com/datasets/nasa/asteroid-impacts)

- Variabel-variabel pada Dataset
    Setelah kedua file (`Impacts.csv` dan `Orbits.csv`) digabungkan, dataset ini memiliki variabel-variabel sebagai berikut:
    - Dari `Impacts.csv`
        - Object Name: Nama atau penanda unik untuk setiap asteroid.
        - Period Start / Period End: Rentang waktu di mana potensi dampak dipantau.
        - Possible Impacts: Jumlah kemungkinan dampak yang terdeteksi selama periode pemantauan.
        - Cumulative Impact Probability: Akumulasi probabilitas terjadinya dampak dari semua kemungkinan.
        - Asteroid Velocity: Kecepatan asteroid relatif terhadap Bumi saat terjadi potensi dampak (dalam km/s).
        - Asteroid Magnitude: Kecerahan absolut asteroid, yang berhubungan terbalik dengan ukurannya.
        - Asteroid Diameter (km): Perkiraan diameter asteroid dalam kilometer.
        - Cumulative Palermo Scale: Akumulasi nilai Skala Palermo, yang mengukur risiko dampak secara logaritmik.
        - Maximum Palermo Scale: Nilai maksimum Skala Palermo yang pernah tercatat untuk asteroid tersebut.
        - Maximum Torino Scale: Nilai maksimum Skala Torino, skala 0-10 yang mengkategorikan bahaya dampak.
    - Dari `Orbits.csv`
        - Object Classification: Kategori orbit asteroid (misalnya, Aten, Apollo, Amor).
        - Orbit Axis (AU): Setengah sumbu utama orbit asteroid dalam satuan astronomi (AU).
        - Orbit Eccentricity: Ukuran kelonjongan orbit asteroid (0 untuk lingkaran sempurna).
        - Orbit Inclination (deg): Kemiringan orbit asteroid terhadap bidang ekliptika (bidang orbit Bumi) dalam derajat.
        - Perihelion Distance (AU): Jarak terdekat asteroid dari Matahari.
        - Aphelion Distance (AU): Jarak terjauh asteroid dari Matahari.
        - Orbital Period (yr): Waktu yang dibutuhkan asteroid untuk satu kali mengorbit Matahari (dalam tahun).
        - Minimum Orbit Intersection Distance (AU): Jarak minimum antara orbit asteroid dan orbit Bumi. Ini adalah salah satu indikator risiko tabrakan yang paling penting.

### Exploratory Data Analysis (EDA)
1. Distribusi Diameter Asteroid
Visualisasi distribusi `Asteroid Diameter (km)` menggunakan histogram menunjukkan bahwa sebagian besar asteroid yang terdeteksi memiliki diameter yang relatif kecil (di bawah 0.5 km). Namun, terdapat beberapa asteroid dengan ukuran yang jauh lebih besar (outliers) yang perlu diwaspadai karena potensi dampaknya yang lebih merusak.
2. Perbandingan Karakteristik Asteroid Berbahaya vs. Tidak Berbahaya
Dengan menggunakan target `is_hazardous` yang telah dibuat, kita bisa membandingkan karakteristik kedua kelompok. Misalnya, visualisasi box plot untuk `Minimum Orbit Intersection Distance (MOID)` menunjukkan bahwa kelompok asteroid yang diklasifikasikan sebagai "Berbahaya" memiliki nilai MOID yang secara signifikan lebih rendah dan lebih terkonsentrasi di dekat nol. Ini mengonfirmasi bahwa jarak orbit yang sangat dekat dengan orbit Bumi adalah faktor kunci dari sebuah ancaman.
3. Matriks Korelasi Antar Fitur
Untuk memahami hubungan antar variabel numerik, dibuat sebuah heatmap dari matriks korelasi. Dari heatmap ini, terlihat adanya korelasi positif yang kuat antara `Orbit Axis` dan `Orbital Period`, yang sesuai dengan Hukum Kepler Ketiga. Selain itu, terlihat korelasi negatif antara `Asteroid Magnitude` dan `Asteroid Diameter`, yang mengonfirmasi bahwa asteroid yang lebih terang (magnitudo lebih kecil) cenderung berukuran lebih besar.

## Data Preparation
Tahap Data Preparation adalah salah satu proses paling krusial dalam proyek machine learning. Tujuannya adalah untuk membersihkan, mengubah, dan menstrukturkan data mentah agar siap digunakan untuk proses permodelan. Data yang bersih dan terstruktur dengan baik akan menghasilkan model yang lebih akurat dan andal.
Berikut adalah tahapan-tahapan yang dilakukan secara berurutan dalam proyek ini.
1. Penggabungan Data (Data Merging)
    - Proses: Langkah pertama adalah memuat kedua file dataset, `Impacts.csv` dan `Orbits.csv`, ke dalam dua DataFrame terpisah menggunakan library pandas. Setelah itu, kedua DataFrame tersebut digabungkan menjadi satu DataFrame tunggal menggunakan kolom `'Object Name'` sebagai kunci (key) penggabungan.
    - Alasan: Data yang dibutuhkan untuk proyek ini terpisah di dua file berbeda. `Impacts.csv` berisi informasi terkait risiko dampak (calon target), sementara `Orbits.csv` berisi parameter-parameter orbit yang akan menjadi fitur prediksi. Menggabungkan keduanya adalah langkah esensial untuk menyatukan semua informasi yang relevan untuk setiap asteroid dalam satu tabel.

2. Penanganan Nilai Hilang (Handling Missing Values)
    - Proses: Setelah data digabungkan, dilakukan pengecekan untuk mengidentifikasi adanya nilai yang hilang (missing values atau NaN) di setiap kolom menggunakan fungsi `.isnull().sum()`. Baris data yang mengandung nilai hilang kemudian dihapus dari dataset.
    - Alasan: Sebagian besar algoritma machine learning tidak dapat bekerja dengan data yang hilang. Menghapus baris yang tidak lengkap adalah strategi yang cepat dan aman, terutama jika jumlah data yang hilang relatif kecil dibandingkan total dataset. Hal ini untuk memastikan bahwa model hanya dilatih menggunakan data yang lengkap dan berkualitas.

3. Rekayasa Fitur (Feature Engineering) - Pembuatan Kolom Target
    - Proses: Ini adalah tahap kunci untuk masalah klasifikasi. Sebuah kolom baru bernama `is_hazardous` dibuat sebagai variabel target. Kolom ini diisi dengan nilai biner (1 atau 0) berdasarkan kondisi dari kolom `Maximum Torino Scale`. Jika nilai `Maximum Torino Scale` lebih besar dari 0, maka `is_hazardous` diberi nilai 1 (Berbahaya). Jika tidak, nilainya adalah 0 (Tidak Berbahaya).
    - Alasan: Dataset asli tidak menyediakan label klasifikasi langsung. Tahapan ini menerjemahkan masalah bisnis ("Apakah asteroid ini berbahaya?") menjadi target teknis yang dapat dipahami dan diprediksi oleh model klasifikasi. Pemilihan `Maximum Torino Scale` sebagai acuan didasarkan pada definisinya sebagai skala kategoris untuk bahaya dampak.

4. Pemilihan Fitur (Feature Selection)
    - Proses: Beberapa kolom yang tidak relevan atau berisiko menyebabkan kebocoran data (data leakage) dihapus. Kolom-kolom yang dihapus antara lain: 'Object Name' (hanya sebagai identifier), 'Period Start', 'Period End' (informasi non-numerik yang tidak relevan), serta Maximum Palermo Scale dan `Maximum Torino Scale` (karena menjadi sumber dari variabel target).
    - Alasan: Tujuannya adalah untuk menyederhanakan model dan hanya menggunakan fitur-fitur yang benar-benar prediktif. Menghapus kolom identifier dan teks yang tidak relevan akan meningkatkan efisiensi. Yang terpenting, menghapus kolom asli yang menjadi dasar pembuatan target (`Maximum Torino Scale`) adalah wajib untuk mencegah data leakage, yaitu kondisi di mana model "mencontek" jawaban saat pelatihan, sehingga menghasilkan performa yang sangat tinggi namun tidak realistis pada data baru.

5. Pembagian Dataset (Train-Test Split)
    - Proses: Dataset yang sudah bersih dibagi menjadi dua bagian: fitur (X) yang berisi semua kolom prediktor, dan target (y) yang berisi kolom `is_hazardous`. Selanjutnya, data ini dibagi lagi menjadi data latih (training set) dan data uji (testing set) dengan proporsi 80% untuk latih dan 20% untuk uji. Parameter `stratify` digunakan pada variabel target (y) untuk memastikan proporsi kelas berbahaya dan tidak berbahaya sama di kedua set.
    - Alasan: Pembagian ini krusial untuk evaluasi model yang objektif. Model akan "belajar" dari training set, dan kemudian kemampuannya untuk menggeneralisasi pada data yang belum pernah dilihat sebelumnya akan diuji pada testing set. Penggunaan `stratify` sangat penting karena jumlah asteroid berbahaya jauh lebih sedikit (kelas minoritas), sehingga memastikan kedua set data memiliki representasi kelas yang seimbang.

6. Penskalaan Fitur (Feature Scaling)
    - Proses: Semua fitur numerik dalam data latih dan data uji diskalakan menggunakan StandardScaler dari library scikit-learn. Proses fit_transform diterapkan pada data latih, dan proses transform diterapkan pada data uji.
    - Alasan: Fitur-fitur dalam dataset ini memiliki rentang nilai yang sangat berbeda (misalnya, Asteroid Diameter vs. Orbit Eccentricity). Algoritma seperti Logistic Regression sangat sensitif terhadap skala fitur. Penskalaan (standardisasi) memastikan semua fitur memiliki skala yang sebanding, sehingga tidak ada satu fitur pun yang mendominasi proses pembelajaran model hanya karena rentang nilainya yang besar.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

