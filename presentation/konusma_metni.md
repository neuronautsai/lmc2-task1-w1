# Konuşma metni: Review sonrası durum

Light My Cells 2, Task 1 sunumunun sayfa sayfa konuşma metni.

## 1. Review sonrası durum

Merhaba. Bu sunumda Light My Cells 2 challenge’ının birinci görevi üzerine yaptığım çalışmanın son durumunu anlatacağım. Week 3 sonunda modelin temel hali hazırdı ve beş iyileştirme maddesi belirlenmişti. Bugün bu beş işten dördünün bittiğini, birinin hâlâ sürdüğünü göreceksiniz. Her iş için ne yaptığımı, ne çıktığını ve buna göre ne karar verdiğimi sırayla göstereceğim.

## 2. Boyamadan floresan görüntü elde etmek

Önce challenge’ın neden var olduğunu anlatayım. Hücrelerde çekirdek, mitokondri gibi yapıları görmek için genelde floresan mikroskopi kullanılıyor. Ama bunun için hücreleri boyamak gerekiyor. Bu hem pahalı, hem zaman alıyor, hem de canlı hücreleri etkileyebiliyor. Işık mikroskobu ise boyasız çalışıyor ama bu yapıları net göstermiyor. Challenge’ın amacı, boyasız ışık görüntüsünden floresan görüntüyü yapay zekayla tahmin etmek. Buna in-silico labeling deniyor. Yarışmayı France-BioImaging düzenliyor ve iki görev var. Ben birinci görevde çalışıyorum: tek bir ışık görüntüsünden dört yapının en net odaktaki floresan görüntüsünü tahmin etmek.

## 3. Final değerlendirme 1 ile 13 Aralık arasında

Takvime de kısaca bakalım. Şu an validation aşamasındayız. Bu aşama 30 Kasım’da bitiyor ve en fazla beş deneme submission hakkı veriyor, bizim üç hakkımız kaldı. Mock aşaması 1 Aralık gece 01:59’da, İstanbul saatiyle kapanıyor. Asıl değerlendirme 1 ile 13 Aralık arasında ve bu aşamada sadece tek bir final submission yapılabiliyor. Sonuçlar 18 Aralık’ta açıklanacak. Bu yüzden kalan işleri, final submission’dan önce en iyi modeli seçebilecek şekilde sıraladım.

## 4. Işık mikroskobu görüntüsünden floresan kanalları

Görevi bir örnekle göstereyim. Soldaki görüntü modelin girdisi, boyasız bir brightfield görüntüsü. Ortada ve sağda aynı alanın floresan görüntüleri var: mitokondri ve çekirdek. Model soldaki görüntüye bakıp bu kanalları üretmeye çalışıyor. Her görüntüde dört kanalın hepsi bulunmuyor, bu yüzden model sadece var olan kanallarla eğitiliyor ve ölçülüyor. Kullandığım model bir U-Net, yani girdisi de çıktısı da görüntü olan bir sinir ağı.

## 5. Üç ölçü, iki değerlendirme yolu

Sonuçları okumadan önce kullandığım üç ölçüyü kısaca anlatayım. SSIM, tahmin edilen görüntü ile gerçek görüntünün yapı olarak ne kadar benzediğini ölçüyor, 1 mükemmel uyum demek. PCC, parlak ve karanlık bölgelerin aynı yerlerde olup olmadığına bakıyor, o da 1’e yaklaştıkça iyi. MAE ise piksel başına ortalama hata, düşük olması iyi. Portal takımları bu ölçülerdeki sıralamaların ortalamasıyla sıralıyor. Benim hızlı yerel skorum iki denemede yanlış yönü gösterdi. Bu yüzden kararları, portalın ölçüm kuralını yerelde taklit eden portal metriğiyle veriyorum. Mock ise portalda iki görüntüyle yapılan gerçek bir deneme.

## 6. Beş işin genel durumu

Beş işin genel tablosu bu. İlk iş verideki hataları kontrol etmekti, bitti ve kopya dosyaları çıkardım. İkinci iş çıktının yoğunluk aralığıydı, portalda skorlar düştüğü için eski sürüme döndüm. Üçüncü iş görüntü parçalarını birleştirme yöntemiydi, bantlar azaldı. Dördüncü iş hedef normalizasyonuydu, iyileşme olmadı. Beşinci iş eğitimde SSIM’i doğrudan hedeflemekti, hâlâ sürüyor ve ilk sonuç olumsuz. Olumsuz sonuçları da kayıt altına aldım, çünkü hangi yolun işe yaramadığını bilmek de bir sonuç.

## 7. Study 3’te kopya dosyalar bulundu

İlk iş verideki hataları bulmaktı. Her görüntü için her organelin ayrı bir dosyası var. Aynı dosyanın yanlışlıkla iki farklı organel olarak kaydedilip kaydedilmediğini kontrol ettim. Study 3’te dört görüntünün Actin dosyası, Nucleus dosyasının piksel piksel aynısı çıktı. Yani model bu görüntülerde Actin yerine aslında Nucleus ile ölçülüyordu. Bu görüntülerin hepsi validation setindeydi, eğitim etkilenmedi. Study 3’ün tüm Actin hedeflerini çıkardım. Skorlar çok az değişti ve yöntemlerin sıralaması aynı kaldı. Bundan sonraki bütün deneyleri bu temiz listeyle yaptım.

## 8. Portalda skorlar düştü, eski sürüme döndük

İkinci iş modelin çıktısını hangi yoğunluk aralığında yazdığımızla ilgiliydi. Model şu an çıktıyı 0 ile 255 arasında yazıyor. Gerçek mikroskop görüntüleri ise binlerce değere kadar çıkıyor. Portaldaki hata ölçülerinin çok büyük görünmesinin sebebi bu fark olabilir diye düşündüm. Bu yüzden çıktıyı gerçek yoğunluk aralığında yazan bir sürüm hazırladım. Yerel testte ortalama hata yüzde 14 azaldı. Ama portaldaki Mock denemesinde skorlar düştü, örneğin Nucleus PCC 0.435’ten 0.382’ye indi. Demek ki portal hatayı benim varsaydığım şekilde hesaplamıyor. Eski sürüme döndüm ve portalın hesaplama yöntemini öğrenmeden tekrar denemeyeceğim. Beş submission hakkımızdan üçü kaldı.

## 9. Parça sınırlarındaki bantlar azaldı

Üçüncü iş görüntüleri birleştirme yöntemiydi. Büyük görüntüler 512 piksellik parçalar halinde tahmin edilip sonra birleştiriliyor. Parçaların birleştiği yerlerde, özellikle neredeyse boş görüntülerde, çizgi gibi bantlar görülüyordu. Dört birleştirme yöntemini karşılaştırdım. Eski yöntem de kenarları yumuşatıyordu ama sınırda küçük bir sıçrama bırakıyordu. Hann penceresi bu sıçramayı neredeyse sıfıra indirdi. Bant ölçüsü 1.45’ten 1.07’ye düştü, 1 hiç bant yok demek. Skor artışı küçük ama tutarlı. Hann’ı varsayılan yapmayı öneriyorum, bunun için ayrı submission harcamıyorum.

## 10. Portalın aralığıyla eğitmek iyileştirmedi

Dördüncü iş hedef normalizasyonuydu. Eğitimde gerçek floresan görüntüleri belirli bir aralığa ölçekleyip en parlak noktaları kırpıyordum. Portal ise değerlendirmede farklı bir aralık kullanıyor ve kırpmıyor. Bu farkın, modelin en parlak bölgeleri öğrenmesini engelleyip engellemediğini test ettim. Eğitim hedeflerini portalın aralığıyla ölçekledim ve kırpmayı kaldırdım. Bunun çalışması için modelin çıkışını da sınırsız yapmam gerekti. Yerel skor iyileşme gösterdi ama portal metriği göstermedi, SSIM 0.437’den 0.413’e düştü. Bu yüzden eski normalizasyonla devam ediyorum. Bu deney yerel skora tek başına güvenmememiz gerektiğini de gösterdi.

## 11. İlk sonuçta skorlar düştü

Beşinci iş hâlâ sürüyor. Model normalde sadece piksel hatasını küçültecek şekilde eğitiliyor. Değerlendirmede SSIM önemli olduğu için eğitime SSIM’i doğrudan iyileştiren bir terim ekledim. Sonucun şansa bağlı olmadığını görmek için aynı deneyi üç farklı başlangıçla, yani üç seed ile yapmayı planladım. İlk eğitimde portal metriğindeki bütün skorlar düştü. En büyük düşüş Nucleus PCC’de, 0.659’dan 0.441’e. İkinci eğitim de aynı yönü gösteriyor. Olası sebep, eğitimdeki SSIM’in ham yoğunluklar üzerinden hesaplanması. Portal tahmini kendisi yeniden ölçeklediği için bu kazanç orada kayboluyor olabilir, bunu henüz test etmedim. İkinci eğitimin portal metriğine bakıp üçüncüye karar vereceğim.

## 12. Submission’a kadar dört adım

Önümüzdeki adımlar sırayla şunlar. Önce ikinci eğitimi portal metriğiyle ölçeceğim ve üçüncü eğitimin gerekip gerekmediğine karar vereceğim. Sonra beşinci işin sonucunu issue’ya yazacağım. Ardından şimdiye kadarki en iyi modeli, Hann birleştirme yöntemiyle birlikte submission paketine koyacağım. En son tek bir Mock submission yapacağım. Üç hakkımız kaldı ve Mock aşaması 1 Aralık’ta kapanıyor. Bu hakkı sadece en iyi modelle ve bir kez kullanmak istiyorum. Teşekkürler, sorularınızı alabilirim.
