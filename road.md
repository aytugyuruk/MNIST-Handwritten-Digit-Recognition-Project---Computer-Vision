## Proje 4: El Yazısı Rakam Tanıma (Klasik Makine Öğrenmesi ile)

### Gerçek Hayat Uygulaması
Posta kodlarını okuma, formlardaki sayıları dijitalleştirme gibi Optik Karakter Tanıma (OCR) sistemlerinin en temel hali.

### Neden Senin İçin Uygun?
Bu proje, görüntü işleme ve makine öğrenmesi bilginizi birleştireceğiniz ilk ciddi adım olacak. Derin öğrenme kullanmadan, klasik bir ML modeli (SVM, k-NN, Lojistik Regresyon gibi) ile bu işin nasıl yapılabileceğini göreceksiniz.

### Kullanacağın Temel Teknikler
- MNIST gibi hazır bir veri setini kullanma
- Her bir rakam görüntüsünden özellik çıkarma (feature extraction). (Örn: Görüntüyü düz bir vektöre çevirme (flattening), piksel yoğunluklarını kullanma)
- Scikit-learn kütüphanesini kullanarak bir sınıflandırıcı model eğitme (örn: SVC veya KNeighborsClassifier)
- Eğittiğiniz modeli, daha önce görmediği rakam görüntülerini tahmin etmek için kullanma

**Zorluk Seviyesi:** Orta / İleri