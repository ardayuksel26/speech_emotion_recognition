import { useEffect, useMemo, useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { motion, useScroll, useSpring, AnimatePresence } from 'framer-motion';
import { 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie,
    Area, AreaChart
} from 'recharts';
import { FaGraduationCap, FaNetworkWired, FaServer, FaMicrophoneAlt, FaBrain, FaDatabase, FaCode, FaFlask, FaChevronDown, FaFolderOpen } from 'react-icons/fa';

// Import our auto-generated real-world test results
import { MastermindMetrics } from '../data/realWorldResults';
import { combinedWordBenchmarks } from '../data/wordBenchmarks';
import { sentenceVoskSynthetic15, sentenceVoskHq } from '../data/sentenceBenchmarkCharts';
import { segmentationCatBoostOnly, segmentationQuadMean, vadEnergyIllustration } from '../data/segmentationBenchmark';

const turEvData = [
    { name: "Angry", value: 487, color: "#ef4444" },
    { name: "Sad", value: 483, color: "#6366f1" },
    { name: "Calm", value: 408, color: "#14b8a6" },
    { name: "Happy", value: 357, color: "#f59e0b" }
];

type TestArtifact = { file: string; summary: string; relatesTo: string };

const TEST_ARTIFACTS: TestArtifact[] = [
    { file: 'benchmark_result.txt', summary: 'Kelime düzeyi 5-fold CV; standart TurEV dalgaları üzerinde model başına doğruluk özeti.', relatesTo: 'Bölüm 4.2' },
    { file: 'benchmark_robust_results.txt', summary: 'Aynı protokolün gürültü/enjeksiyon ile güçlendirilmiş (robust) eğitim ağırlıklarıyla tekrarı.', relatesTo: 'Bölüm 4.2 / 4.4' },
    { file: 'word_methods_benchmark_results.txt', summary: 'Kelime benchmark çıktısının özet metin dökümü (ortalama, standart sapma).', relatesTo: 'Bölüm 4' },
    { file: 'sentence_methods_benchmark_results.txt', summary: 'Sentetik cümle (15×4); VAD / VOSK / WhisperX segmentasyonları × tüm modeller; precision/recall/F1 tabloları.', relatesTo: 'Bölüm 4.5, 5.6' },
    { file: 'sentence_hq_methods_benchmark_results.txt', summary: 'HQ cümle (10×4); daha uzun kayıtlar ve mikrofon simülasyonu; aynı metot matrisi.', relatesTo: 'Bölüm 4.5' },
    { file: 'mastermind_benchmark_results.txt', summary: 'Üretim Mastermind hattının cümle düzeyi doğruluk ve sınıf metrikleri (API / sentetik doğrulama).', relatesTo: 'Bölüm 3.5' },
    { file: 'mastermind_confusion_matrix_result.txt', summary: 'Karışıklık matrisi sayısal çıktısı; sınıf karışımını nicel okumak için.', relatesTo: 'Bölüm 3' },
    { file: 'validation_robust_metrics.txt', summary: 'Robust doğrulama koşullarında özet metrikler (literatürle kıyas için).', relatesTo: 'Bölüm 4' },
    { file: 'emotion_performance_metrics.txt', summary: 'Duygu bazlı performans dökümü; rapor üretiminde kullanılan metin çıktısı.', relatesTo: 'Bölüm 3–4' },
    { file: 'benchmark_robust_results.png, benchmark_results.png, …', summary: 'Matplotlib ile üretilen şekiller; makale ve sunum için statik görseller (TestsResults/).', relatesTo: 'Arşiv / tekrarlanabilirlik' },
];

const scrollToTechnicalId = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

const TechnicalInfoPage = () => {
    const { isDark } = useTheme();
    const { scrollYProgress } = useScroll();
    const scaleX = useSpring(scrollYProgress, {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    // Formatting raw text data for Recharts structure
    const barChartData = MastermindMetrics.metrics.map(d => ({
        name: d.emotion,
        Precision: parseFloat((d.precision * 100).toFixed(1)),
        Recall: parseFloat((d.recall * 100).toFixed(1)),
        F1: parseFloat((d.f1 * 100).toFixed(1)),
    }));

    // Scroll to top on mount
    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    // Text color utility specifically for strong/bold tags in dark vs light mode
    const strongClass = isDark ? "text-white font-bold" : "text-slate-900 font-bold";

    const [testHubTab, setTestHubTab] = useState<'ozet' | 'kelime' | 'cumle' | 'segment' | 'artifacts'>('ozet');
    const [sentenceSuite, setSentenceSuite] = useState<'synth' | 'hq'>('synth');
    const [openArtifactIndex, setOpenArtifactIndex] = useState<number | null>(null);
    const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

    const avgSynthSentenceAcc = useMemo(
        () => sentenceVoskSynthetic15.reduce((a, b) => a + b.accStd, 0) / sentenceVoskSynthetic15.length,
        []
    );
    const avgHqSentenceAcc = useMemo(
        () => sentenceVoskHq.reduce((a, b) => a + b.accStd, 0) / sentenceVoskHq.length,
        []
    );
    const wordAvgStandardAcc = useMemo(
        () => combinedWordBenchmarks.reduce((a, b) => a + b.Standard_Accuracy, 0) / combinedWordBenchmarks.length,
        []
    );
    const mastermindMacroF1Mean = useMemo(
        () => MastermindMetrics.metrics.reduce((a, b) => a + b.f1, 0) / MastermindMetrics.metrics.length,
        []
    );

    const testHubTabs: { id: typeof testHubTab; label: string }[] = [
        { id: 'ozet', label: 'Özet' },
        { id: 'kelime', label: 'Kelime CV' },
        { id: 'cumle', label: 'Cümle' },
        { id: 'segment', label: 'Segmentasyon' },
        { id: 'artifacts', label: 'Artefaktlar' },
    ];

    type CumleBarRow = { name: string; accStd: number; accRob: number; f1Std: number; f1Rob: number };
    const cumleChartData: CumleBarRow[] = useMemo(() => {
        const src = sentenceSuite === 'synth' ? sentenceVoskSynthetic15 : sentenceVoskHq;
        return src.map((r) => ({
            name: r.name,
            accStd: r.accStd,
            accRob: r.accRob,
            f1Std: r.f1Std,
            f1Rob: r.f1Rob,
        }));
    }, [sentenceSuite]);

    return (
        <div className={`min-h-[100vh] pb-32 font-sans transition-colors duration-700 w-full overflow-x-hidden ${isDark ? 'bg-[#0a0f1d] text-[#cbd5e1]' : 'bg-[#fafafa] text-[#334155]'}`}>
            {/* Top Reading Progress Bar */}
            <motion.div
                className="fixed top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-fuchsia-500 z-50 origin-left"
                style={{ scaleX }}
            />

            {/* HERO ACADEMIC HEADER */}
            <div className="w-full pt-40 pb-20 px-8 flex justify-center items-center relative overflow-hidden">
                <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] rounded-full blur-[150px] opacity-20 pointer-events-none ${isDark ? 'bg-indigo-600' : 'bg-indigo-200'}`} />
                
                <div className="relative z-10 max-w-[1000px] text-center">
                    <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
                        <div className="flex justify-center mb-6">
                            <FaGraduationCap className="text-6xl text-indigo-500 drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]" />
                        </div>
                        <h1 className={`text-4xl md:text-6xl font-black mb-6 tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            The Science Behind <br /> 
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-fuchsia-500">
                                Mastermind Architecture
                            </span>
                        </h1>
                        <p className={`text-xl font-light leading-relaxed max-w-4xl mx-auto px-4 ${isDark ? 'opacity-80' : 'opacity-70'}`}>
                            Derin Öğrenme, Ensemble Teknikleri, Ses Frekans Analizi ve Otonom Test Laboratuvarının arkasındaki akademik altyapı ve mimari detaylar. Sesten duygu tanımanın (SER) perde arkasına kapsamlı bir bakış.
                        </p>
                    </motion.div>
                </div>
            </div>

            {/* SCROLLING DOCUMENT STRUCTURE - CENTERED */}
            <div className="w-full flex justify-center pb-24 px-4">
                <div className="w-full max-w-[1200px] space-y-20 text-lg md:text-[1.1rem] leading-[2] tracking-wide text-justify">

                {/* 1. GİRİŞ */}
                <motion.section initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaBrain className="text-indigo-500 flex-shrink-0" />
                        1. Projenin Amacı ve Yaklaşımı
                    </h2>
                    <p className="mb-6 font-semibold opacity-90 text-indigo-500 uppercase tracking-wider text-sm">
                        Akademik Tez Başlığı: EMOTION RECOGNITION FROM TURKISH AUDIO IN WORD AND SENTENCE LEVELS
                    </p>
                    <p className="mb-6">
                        Bu projenin temel tezi, Türkçe dilindeki sesli ifadelerin duygu durumlarını sadece genel ses akışı üzerinden değil, <i className={strongClass}>kelime (word) ve cümle (sentence)</i> bazında ayrıştırarak analiz etmektir. Konuşma Duygu Tanıma (SER - Speech Emotion Recognition) alanındaki mevcut küresel literatür genellikle tek boyutlu (sadece İngilizce tabanlı) modeller kullanır. Bizim yaklaşımımız ise, ses sinyalini en ufak anlamlı birimlerine (kelimelere) kadar parçalamak ve yerel (Türkçe) veri setleri kullanılarak eğitilmiş otonom ağlarla ince taneli (fine-grained) tahminler yapmaktır.
                    </p>
                    <p>
                        Uçucu ve kesintisiz (continuous) olan insan sesini doğrudan tek bir ağa göndermek yerine; gelişmiş kelime bölme (segmentation) motorları (VOSK, WhisperX) aracılığıyla cümleleri kelimelere ayırıyor, her bir kelimenin akustik özelliklerini bağımsız olarak çıkartıyor ve <span className={strongClass}>Mastermind (Üst Akıl)</span> ismini verdiğimiz hibrid-topluluk (ensemble) modelimiz ile bu kelimelerin oylarından genel cümlenin duygu haritasını inşa ediyoruz.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-indigo-300' : 'text-indigo-600'}`}>
                        1.1 Sesten Duygu Tanıma (SER) Problemleri ve Çözümler
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-indigo-500/50">
                        Doğal ortamlardaki (in-the-wild) ses verileri üzerinde çalışırken literatürde karşılaşılan temel problemler ve bu projenin getirdiği çözümler şunlardır:
                    </p>
                    <ul className="space-y-4 text-base list-disc list-inside ml-4 mb-6">
                        <li><span className={strongClass}>Diller Arası Fonetik Farklılıklar (Cross-lingual Bias):</span> Yabancı dilde eğitilmiş bir model, Türkçe'deki kelime vurgularını (stress) yanlış anlayıp hatalı duygu sınıflandırması yapabilir. Bu projede, modele bizzat Türkçe kelime ve hecelerden oluşan kütüphanelerle antrenman yaptırılarak fonetik uçurumlar kapatılmıştır.</li>
                        <li><span className={strongClass}>Zaman Sınırları ve Ses Boşlukları (Silence boundaries):</span> İnsanlar genellikle duygusal durumlarda duraklar veya nefes alırlar. Silero VAD ve VOSK tabanlı kesme algoritmalarımız bu sessizlik kısımlarını izole ederek makinenin gürültüyü bir duygu olarak tahmin etmesini (örneğin sessizliği "Üzgün/Sad" olarak sınıflandırma hatasını) önler.</li>
                        <li><span className={strongClass}>Çevresel Gürültü ve Mikrofon Kalitesi:</span> Üst düzey stüdyo frekanslarındaki veri setleri ile eğitilen modeller gerçek dünyada başarısız olur. Buna karşın, yapay gürültü (Data Augmentation) kullanılarak geliştirilen ve "Veto" kararı verebilen <span className={strongClass}>Robust Katmanlarımız</span> sayesinde sistem dış sese olağanüstü dayanıklı hale (Noise Resilient) gelmiştir.</li>
                    </ul>
                </motion.section>

                {/* 2. VERİ SETİ */}
                {/* 2. VERİ SETİ */}
                <motion.section initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaDatabase className="text-fuchsia-500 flex-shrink-0" />
                        2. Veri Seti ve Sentetik Cümle Jenerasyonu
                    </h2>

                    <h3 className={`text-2xl font-bold mt-8 mb-4 ${isDark ? 'text-fuchsia-300' : 'text-fuchsia-600'}`}>
                        2.1 TurEV-DB (Turkish Emotion Voice Database)
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                        Projemizin derin öğrenme çekirdeği dış kaynaklar yerine yerel dil uyumluluğuna sahip <span className={strongClass}>TurEV-DB</span> veri seti üzerine inşa edilmiştir. Bu veri seti, binlerce bağımsız kelimenin duygu etiketleriyle seslendirilmesinden oluşur. Dört ana duygu sınıfı (<span className="text-red-500 font-bold">Angry</span>, <span className="text-teal-500 font-bold">Calm</span>, <span className="text-amber-500 font-bold">Happy</span>, <span className="text-indigo-500 font-bold">Sad</span>) bünyesinde toplam <span className="font-bold underline text-[1.1rem]">1,735</span> adet temiz, kelime bazlı ses (WAV) dosyası bulunmaktadır. Eğitilen algoritmalar .wav dosyalarını değil, arka planda Librosa ile sesten dışarı aktarılmış yapısal (Varyans, MFCC, Pitch) matrisleri okuyarak (.csv formu) eğitilmektedir.
                    </p>

                    {/* PIE CHART FOR TUREV-DB */}
                    <div className="flex flex-col md:flex-row items-center justify-center my-12 gap-8">
                        <div className="w-full md:w-1/2 h-[350px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie data={turEvData} cx="50%" cy="50%" innerRadius={70} outerRadius={120} paddingAngle={5} dataKey="value">
                                        {turEvData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} stroke={isDark ? "#0f172a" : "#ffffff"} strokeWidth={3} />
                                        ))}
                                    </Pie>
                                    <RechartsTooltip 
                                        contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', borderRadius: '12px', border: 'none', color: isDark ? 'white' : 'black', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                                        formatter={(value) => [`${value} Kelime (WAV)`, 'Veri Sayısı']}
                                    />
                                    <Legend wrapperStyle={{ fontSize: '15px' }}/>
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full md:w-1/2">
                           <h4 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>TurEV-DB Eğitim Verisi Dağılımı</h4>
                           <p className="text-base mb-4 opacity-80">Veri setimizdeki 1,735 kelimenin duygu sınıflarına göre dengeli dağılımı (Distribution) yandaki grafikte gösterilmektedir. Bu dengeli yapı, modellerin bir duyguya (Bias) daha yatkın/meyilli olmasını engeller.</p>
                           <ul className="space-y-2 font-bold text-sm">
                               <li className="text-red-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500"></span> Angry: 487 Örnek (%28)</li>
                               <li className="text-indigo-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-indigo-500"></span> Sad: 483 Örnek (%28)</li>
                               <li className="text-teal-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-teal-500"></span> Calm: 408 Örnek (%24)</li>
                               <li className="text-amber-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-amber-500"></span> Happy: 357 Örnek (%20)</li>
                           </ul>
                        </div>
                    </div>

                    <h3 className={`text-2xl font-bold mt-8 mb-4 ${isDark ? 'text-fuchsia-300' : 'text-fuchsia-600'}`}>
                        2.2 Sentetik (Yapay) Cümle Jenerasyon Ağı
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                        Projenin en yenilikçi yanlarından biri, sadece kelime bazlı performansla yetinmeyerek, izole kelimeleri uç uca bağlayıp "Yapay Cümleler" üreten özel bir simülasyon motoru (Generator) kullanılmasıdır. Gerçek hayattaki bir diyaloğu taklit edebilmek için bu kelimelerin aralıklarına matematiksel olarak hesaplanmış "nefes alma aralıkları ve sessizlik boşlukları (silence gaps)" enjekte edilmiştir.
                    </p>
                    <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                        Bu yaklaşımla üretilen uzun yapay cümleler <span className={strongClass}>`Test/sentencevoice`</span> çalışma alanına aktarılarak production testlerinde Mastermind'ı sınıyor. Üretilen toplam <span className="font-bold underline text-[1.1rem]">79 Pürüzsüz Cümle</span> sınıflandırması şu şekildedir: <span className="text-red-500 font-bold">20 Angry</span>, <span className="text-teal-500 font-bold">20 Calm</span>, <span className="text-amber-500 font-bold">20 Happy</span>, ve <span className="text-indigo-500 font-bold">19 Sad</span>. Modellerimiz asla bu birleştirilmiş devasa cümlelerle önceden eğitilmez; bu cümleler yalnızca Mastermind veto mimarisinin kelimeleri bölüp anlam çıkarma yeteneğini ölçen acımasız bir Validasyon (Doğrulama) Sınavıdır.
                    </p>
                </motion.section>

                {/* 3. MASTERMIND YAPISI */}
                <motion.section id="technical-section-3" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaServer className="text-cyan-500 flex-shrink-0" />
                        3. Mastermind (Üst Akıl) Mimarisi
                    </h2>
                    <p className="mb-6">
                        <span className={strongClass}>Mastermind</span>, üretim API&apos;sinde (<code className="text-sm opacity-90">/api/predict_mastermind</code>) tek bir sınıflandırıcıya bırakılmayan, <i>çok kaynaklı olasılık füzyonu</i> ve <i>koşullu veto</i> ile karar veren hibrit bir karar katmanıdır. Amaç, cümle düzeyinde hem <span className={strongClass}>global prosodi</span> (tüm kayıt üzerinden çıkarılan MFCC vb. vektör) hem de <span className={strongClass}>kelime/segment düzeyi</span> tahminlerini aynı çatı altında birleştirerek, yalnızca tabüler modellerin güçlü yönlerinden yararlanmak; ayrıca literatürde sık görülen &quot;sessizlik veya nefesi Sad sanma&quot; halüsinasyonunu <span className={strongClass}>RF_Robust</span> ile denetlemektir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                        3.1 Bileşenler ve uçtan uca veri akışı
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-cyan-500/50">
                        Sistem üç model ailesini zorunlu tutar: <span className={strongClass}>CatBoost</span>, <span className={strongClass}>XGBoost</span> ve <span className={strongClass}>RF_Robust</span>. Akış şu sırayla işler: (1) yüklenen WAV üzerinden <span className={strongClass}>global özellik vektörü</span> çıkarılır; (2) <span className={strongClass}>VOSK</span> ile kelime zaman damgaları alınır ve her segment için ayrı özellik vektörü üretilir — segment bulunamazsa tüm dosya tek blok gibi ele alınır; (3) her tabüler model için sınıf olasılıkları hem global hem segment ortalamasından hesaplanıp harmanlanır; (4) CatBoost ve XGBoost skorları <i>aritmetik ortalama</i> ile birleştirilir; (5) RF_Robust&apos;un &quot;sad&quot; olasılığı, aşağıdaki eşik kurallarıyla nihai etiketi onaylar, düzeltir veya veto eder. Zaman çizelgesi (timeline) için kelime bazlı hızlı tahminler ayrıca CatBoost ile üretilir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                        3.2 Matematiksel çerçeve: global–segment olasılık harmanı
                    </h3>
                    <p className="mb-4">
                        Dört duygu sınıfı kümesi <span className="font-mono text-[0.92em]">K = {'{'}angry, calm, happy, sad{'}'}</span> (eğitimde varsa &quot;neutral&quot; dışlanarak normalize edilir) için, her model <span className={strongClass}>m</span> ve sınıf <span className={strongClass}>k</span> göz önüne alındığında:
                    </p>
                    <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base leading-relaxed overflow-x-auto ${isDark ? 'bg-slate-900/80 border border-cyan-500/20 text-cyan-100/95' : 'bg-slate-50 border border-cyan-200/80 text-slate-800'}`}>
                        <p className="mb-3"><span className="opacity-70">Global olasılık vektörü:</span> <strong>P</strong><sup>(g)</sup> = softmax veya <strong>predict_proba</strong>(<strong>x</strong><sub>global</sub>)</p>
                        <p className="mb-3"><span className="opacity-70">S segment için ortalama:</span> <strong>P</strong><sup>(seg)</sup> = (1/S) Σ<sub>s=1..S</sub> <strong>P</strong>(<strong>x</strong><sub>s</sub>)</p>
                        <p className="mb-0"><span className="opacity-70">Harmanlanmış dağılım (backend sabiti):</span> <strong>p̂</strong><sup>(m)</sup> = 0.60 · <strong>P</strong><sup>(g)</sup> + 0.40 · <strong>P</strong><sup>(seg)</sup></p>
                    </div>
                    <p className="mb-6 pl-4 border-l-4 border-cyan-500/50">
                        <strong>p̂</strong><sup>(m)</sup> üzerinde, yalnızca hedef duygu sınıflarına ait bileşenler alınır ve toplamı 100&apos;e ölçeklenir: <span className="font-mono text-[0.9em]">score<sub>k</sub> = 100 · p̂<sub>k</sub> / Σ<sub>j∈K</sub> p̂<sub>j</sub></span>. Bu adım, farklı kodlayıcı çıktılarını arayüzde karşılaştırılabilir yüzde skorlarına dönüştürür. Ağırlıkların <span className={strongClass}>α = 0.60</span> (global) ve <span className={strongClass}>β = 0.40</span> (segment ortalaması) seçimi, deneysel cümle değerlendirmelerinde elde edilen doğruluk–F1 dengesine göre sabitlenmiştir; dinamik eşik denemeleri lehine çıkmamıştır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                        3.3 Çift model konsensusu: CatBoost ⊕ XGBoost
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-cyan-500/50">
                        İki güçlü gradient boosting sınıflandırıcısının skorları, sınıf başına ortalanır:
                    </p>
                    <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base ${isDark ? 'bg-slate-900/80 border border-purple-500/20 text-purple-100/95' : 'bg-indigo-50 border border-indigo-200/80 text-slate-800'}`}>
                        main<sub>k</sub> = ( score<sub>k</sub><sup>(CatBoost)</sup> + score<sub>k</sub><sup>(XGBoost)</sup> ) / 2
                    </div>
                    <p className="mb-6">
                        Ön karar <span className="font-mono text-[0.9em]">primary = argmax<sub>k</sub> main<sub>k</sub></span>. Bu yapı, tek bir ağacımsı öğrenicideki aşırı özgüveni yumuşatır; Angry ve Happy gibi sınıflarda iki modelin birbirini tamamladığı gözlemlerine dayanır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                        3.4 RF_Robust: Sad vetosu ve de-hallüsinasyon eşikleri
                    </h3>
                    <p className="mb-4">
                        RF_Robust aynı harmanlama formülüyle skor üretir; Mastermind bunu özellikle <span className={strongClass}>sad</span> kanalının güvenilirliği için kullanır. <span className="font-mono text-[0.9em]">sad<sub>RF</sub></span>, RF skorlarındaki sad yüzdesidir. Karar kuralları (backend ile birebir):
                    </p>
                    <ul className="space-y-3 mb-6 text-base list-disc list-inside ml-2">
                        <li><span className={strongClass}>Kural 1 (Sad reddi / ikinci en iyi):</span> Eğer <span className="font-mono">primary = sad</span> ve <span className="font-mono">sad<sub>RF</sub> &lt; 30</span>, ana modellerin &quot;sad&quot; kararı RF onayı almadığı için düşürülür; nihai sınıf, <span className="font-mono">main</span> skorlarından sad çıkarıldıktan sonra kalanlar içinde maksimum olana atanır (sessizlik/nefeste yanlış üzüntü baskısını azaltır).</li>
                        <li><span className={strongClass}>Kural 2 (Veto):</span> Eğer <span className="font-mono">primary ≠ sad</span> ve <span className="font-mono">sad<sub>RF</sub> ≥ 30</span>, nihai duygu <span className="font-mono">sad</span> ilan edilir ve güven <span className="font-mono">sad<sub>RF</sub></span> ile raporlanır; bu durumda <span className="font-mono">veto_applied = true</span> döner — yani RF, üzgünlük konusunda güçlü kanıt sunduğunda ana konsensusu ezer.</li>
                    </ul>
                    <p className="mb-6 pl-4 border-l-4 border-cyan-500/50 opacity-95">
                        Bu iki kural birlikte, tek başına olasılık çıktısına göre çalışan bir sisteme kıyasla <i>neden–sonuç</i> olarak yorumlanabilir: düşük <span className="font-mono">sad<sub>RF</sub></span> iken sad seçmek istatistiksel olarak güvenilmez kabul edilir; yüksek <span className="font-mono">sad<sub>RF</sub></span> ise üzüntü sinyalini güvenlik ağı olarak öne alır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                        3.5 Deneysel doğrulama: sentetik cümleler ve sınıf metrikleri
                    </h3>
                    <p className="mb-6">
                        Mastermind benzeri hibrit akışın cümle düzeyinde davranışı, TurEV-DB kelimelerinden üretilen <span className={strongClass}>yapay cümle</span> test kümesi ve üretim API yolu üzerinden ölçülmüştür. Aşağıdaki özet, <span className={strongClass}>makro düzeydeki doğruluk</span> ile duygu bazlı <span className={strongClass}>Precision, Recall ve F1</span> değerlerini gösterir; F1 özellikle sınıf dengesizliğinde saf doğruluktan daha bilgilendiricidir.
                    </p>

                    <div className="flex flex-col items-center justify-center mb-10 mt-2 text-center">
                        <span className="text-sm md:text-base font-bold uppercase tracking-widest text-emerald-500 mb-2 opacity-90">
                            Genel Mastermind (Cümle) Accuracy Oranı
                        </span>
                        <span className={`text-6xl md:text-7xl font-black tracking-tighter drop-shadow-lg ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            %{(MastermindMetrics.accuracy * 100).toFixed(1)}
                        </span>
                        <p className={`mt-4 text-sm md:text-base max-w-2xl mx-auto opacity-80 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Duygu Bazlı Model Hassasiyeti (F1, Precision, Recall)
                        </p>
                    </div>

                    <div className={`w-full min-h-[480px] h-[520px] md:h-[560px] py-4 px-2 sm:px-4 md:px-6 rounded-[2.5rem] ${isDark ? '' : ''}`}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={barChartData}
                                margin={{ top: 16, right: 16, left: 8, bottom: 24 }}
                                barCategoryGap="18%"
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? "#334155" : "#e2e8f0"} vertical={false} />
                                <XAxis dataKey="name" stroke={isDark ? "#94a3b8" : "#64748b"} tick={{ fontSize: 14, fontWeight: 'bold' }} />
                                <YAxis
                                    width={56}
                                    stroke={isDark ? "#94a3b8" : "#64748b"}
                                    domain={[0, 100]}
                                    tickFormatter={(val) => `${val}%`}
                                    tick={{ fontSize: 12, fontWeight: 600 }}
                                    tickMargin={10}
                                    dx={-2}
                                />
                                <RechartsTooltip
                                    cursor={{ fill: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' }}
                                    contentStyle={{
                                        backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white',
                                        border: isDark ? '1px solid #334155' : '1px solid #e2e8f0',
                                        borderRadius: '16px',
                                        boxShadow: '0 4px 30px rgba(0,0,0,0.1)',
                                    }}
                                    itemStyle={{ fontSize: '14px', fontWeight: 'bold' }}
                                />
                                <Legend wrapperStyle={{ fontSize: '15px', paddingTop: '24px' }} />
                                <Bar dataKey="F1" name="F1-Score (%)" fill={isDark ? '#a855f7' : '#9333ea'} radius={[6, 6, 0, 0]} />
                                <Bar dataKey="Precision" name="Precision (%)" fill={isDark ? '#3b82f6' : '#2563eb'} radius={[6, 6, 0, 0]} />
                                <Bar dataKey="Recall" name="Recall (%)" fill={isDark ? '#ec4899' : '#db2777'} radius={[6, 6, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <p className={`text-sm text-center max-w-3xl mx-auto mt-2 mb-2 opacity-75 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                        Sad sınıfında düşük recall, üzüntünün akustik olarak diğer sınıflarla karışması ve veto/de-hallüsinasyon dengesiyle uyumludur; Angry tarafında yüksek precision, modelin öfkeyi yanlış pozitif saymada muhafazakâr kaldığını gösterir.
                    </p>
                </motion.section>

                {/* 4. SINIFLANDIRMA MODELLERİ */}
                <motion.section id="technical-section-4" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaFlask className="text-amber-500 flex-shrink-0" />
                        4. Sınıflandırma Modelleri (10+ Model Anatomisi)
                    </h2>
                    <p className="mb-8">
                        Bu bölümde, TurEV-DB üzerinde eğitilen tüm sınıflandırıcılar <span className={strongClass}>ortak bir özellik uzayı</span> (Librosa / OpenSMILE tabanlı, backend&apos;de yaklaşık <span className={strongClass}>1584 boyutlu</span> vektör; CNN/DNN için gerekli tensör yeniden şekillendirmeleriyle) paylaşır. Fark, <i>öğrenen fonksiyonun parametreleştirilmesi</i> ve <i>optimizasyon hedefidir</i>. Kelime düzeyinde değerlendirme, etiketli izole kelime dalgalarının <span className={strongClass}>5-fold çapraz doğrulama</span> ile ayrılmasına dayanır; cümle düzeyinde ise sentetik cümle üretiminden gelen uzun kayıtlar segmente edilip (VOSK) Bölüm 3&apos;te özetlenen olasılık harmanı veya doğrudan sınıflandırıcı çıktıları raporlanır. Aşağıdaki alt başlıklar hem kuramsal hem de <code className="text-sm opacity-90">Test/TestsResults</code> klasöründeki ölçümlerle uyumludur.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.1 Denetimli öğrenme çerçevesi ve sınıflandırma metrikleri
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                        Çok sınıflı duygu problemi, öğrenme çiftleri <span className="font-mono text-[0.88em]">(x<sub>i</sub>, y<sub>i</sub>)</span> ile kurulur; <span className="font-mono">x<sub>i</sub> ∈ ℝ<sup>d</sup></span> akustik özellik vektörü, <span className="font-mono">y<sub>i</sub> ∈ {'{'}1,…,C{'}'}</span> etikettir (<span className="font-mono">C = 4</span>: Angry, Calm, Happy, Sad). Bir sınıflandırıcı <span className="font-mono">f</span> tahmin <span className="font-mono">ŷ = f(x)</span> üretir; olasılıksal modellerde <span className="font-mono">p<sub>k</sub>(x) = P(y = k | x)</span> softmax veya ağaç tabanlı <span className="font-mono">predict_proba</span> ile elde edilir.
                    </p>
                    <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base leading-relaxed overflow-x-auto ${isDark ? 'bg-slate-900/80 border border-amber-500/20 text-amber-100/95' : 'bg-amber-50/90 border border-amber-200/80 text-slate-800'}`}>
                        <p className="mb-2">Sınıf <span className="font-mono">k</span> için: Precision<sub>k</sub> = TP<sub>k</sub> / (TP<sub>k</sub> + FP<sub>k</sub>), Recall<sub>k</sub> = TP<sub>k</sub> / (TP<sub>k</sub> + FN<sub>k</sub>)</p>
                        <p className="mb-0">F1<sub>k</sub> = 2 · Precision<sub>k</sub> · Recall<sub>k</sub> / (Precision<sub>k</sub> + Recall<sub>k</sub>) &nbsp;·&nbsp; Macro-F1 = (1/C) Σ<sub>k</sub> F1<sub>k</sub></p>
                    </div>
                    <p className="mb-6 pl-4 border-l-4 border-amber-500/50">
                        <span className={strongClass}>Accuracy</span>, doğru tahmin oranıdır; dengesiz sınıflarda yanıltıcı olabileceği için cümle raporlarında <span className={strongClass}>macro-F1</span> birlikte verilir. Kelime CV&apos;de yalnızca doğruluk raporu üretilmiştir (<code className="text-xs opacity-80">benchmark_result.txt</code>, <code className="text-xs opacity-80">benchmark_robust_results.txt</code>). <span className={strongClass}>K-fold CV</span> için model, veriyi <span className="font-mono">K</span> parçaya böler; her bölüm sırayla doğrulama, kalan <span className="font-mono">K−1</span> parça eğitim olur; raporlanan skor <span className="font-mono">(1/K) Σ<sub>j</sub> Acc<sub>j</sub></span> ortalamasıdır (<span className="font-mono">K = 5</span>).
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.2 Kelime düzeyi (TurEV-DB): tabüler boosting ve baseline modeller
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                        <span className={strongClass}>Gradient boosting</span> (CatBoost, XGBoost, LightGBM, sklearn GradientBoosting), aşamalı olarak eklenen zayıf öğreniciler <span className="font-mono">h<sub>t</sub></span> ile toplu tahmin <span className="font-mono">F<sub>T</sub>(x) = Σ<sub>t=1..T</sub> η<sub>t</sub> h<sub>t</sub>(x)</span> kurar; kayıp genellikle log-loss (çok sınıf) veya benzeri türevlenebilir bir fonksiyon üzerinden azaltılır. Ağaçlar, özellik uzayında eksene paralel bölmelerle <i>yerel karar sınırları</i> öğrenir; bu, MFCC türevleri ve spektral kontrast gibi doğrusal olmayan ayrışmalara elverişlidir. <span className={strongClass}>Random Forest</span>, bootstrap alt örneklerde eğitilmiş ağaçların oylaması <span className="font-mono">ŷ = mode<sub>t</sub> h<sub>t</sub>(x)</span> ile varyansı düşürür. <span className={strongClass}>SVM</span> (RBF çekirdek), <span className="font-mono">max<sub>α</sub> Σ α<sub>i</sub> − (1/2) Σ<sub>i,j</sub> α<sub>i</sub>α<sub>j</sub> y<sub>i</sub>y<sub>j</sub> K(x<sub>i</sub>,x<sub>j</sub>)</span> çiftini çözer; <span className={strongClass}>k-NN</span> ise <span className="font-mono">ŷ = maj(</span><span className="font-mono">{'{'}</span><span className="font-mono">y<sub>j</sub> : x<sub>j</sub> ∈ N<sub>k</sub>(x)</span><span className="font-mono">{'}'}</span><span className="font-mono">)</span> ile Öklid komşuluğuna dayanır — düşük gecikme için referans taban çizgidir.
                    </p>
                    <p className="mb-6 text-base opacity-90">
                        Aşağıdaki grafik, <span className={strongClass}>standart TurEV kelime dalgaları</span> ile <span className={strongClass}>gürültü/enjeksiyon ile zenginleştirilmiş (robust)</span> kopyalar üzerindeki 5-fold doğruluk ortalamalarını karşılaştırır. En yüksek çift skor CatBoost&apos;ta gözlenir; bu nedenle Mastermind ana motorları CatBoost ve XGBoost arasında seçilmiştir.
                    </p>
                    <div className="w-full min-h-[400px] h-[440px] py-4 px-1 sm:px-4 rounded-[2rem]">
                        <h4 className={`text-center font-bold text-xs sm:text-sm tracking-widest uppercase mb-6 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            Kelime: 5-fold CV doğruluğu (%) — benchmark_result.txt &amp; benchmark_robust_results.txt
                        </h4>
                        <ResponsiveContainer width="100%" height="88%">
                            <BarChart data={combinedWordBenchmarks} margin={{ top: 12, right: 18, left: 10, bottom: 8 }} barCategoryGap="14%">
                                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-28} textAnchor="end" height={68} />
                                <YAxis width={54} stroke={isDark ? '#94a3b8' : '#64748b'} domain={[60, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fontWeight: 600 }} tickMargin={8} />
                                <RechartsTooltip
                                    cursor={{ fill: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' }}
                                    contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', border: isDark ? '1px solid #334155' : '1px solid #e2e8f0', borderRadius: '14px' }}
                                    itemStyle={{ fontSize: '13px', fontWeight: 'bold' }}
                                />
                                <Legend wrapperStyle={{ fontSize: '13px', paddingTop: '12px' }} />
                                <Bar dataKey="Standard_Accuracy" name="Standart TurEV (%)" fill={isDark ? '#d97706' : '#f59e0b'} radius={[5, 5, 0, 0]} />
                                <Bar dataKey="Robust_Accuracy" name="Robust / gürültülü (%)" fill={isDark ? '#7c3aed' : '#8b5cf6'} radius={[5, 5, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <h3 className={`text-2xl font-bold mt-14 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.3 Derin modeller: MLP (DNN) ve 1D-CNN
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                        <span className={strongClass}>MLP</span> için katmanlar <span className="font-mono">h<sup>(ℓ)</sup> = σ(W<sup>(ℓ)</sup>h<sup>(ℓ−1)</sup> + b<sup>(ℓ)</sup>)</span> biçiminde ilerler; çıkışta softmax veya logits üzerinden çapraz entropi minimize edilir. Tabüler özellik vektörü doğrudan tam bağlı katmanlara beslendiğinde model, ağaçlara kıyasla daha pürüzsüz ama bazen daha fazla veri ihtiyacı duyan sınırlar öğrenir. <span className={strongClass}>1D-CNN</span>, önce özellik dizisini <span className="font-mono">(batch, zaman veya kanal, 1)</span> şekline getirip konvolüsyon çekirdekleriyle yerel spektral-zamansal kalıpları yakalar; havuzlama ile öteleme dayanıklılığı artar. Kelime CV&apos;de MLP orta düzeyde; cümle senaryosunda segmentasyon ve sınıf dengesizliği yüzünden tabüler boosting genelde daha stabil kalır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.4 Robust eğitim varyantları (*_robust)
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-amber-500/50">
                        <span className="font-mono">*_robust</span> modelleri, eğitim verisine kontrollü gürültü / bozunum eklenerek elde edilen paralel ağırlıklardır. Amaç, test dağılımı gerçek dünya koşullarına kaydığında <span className={strongClass}>genelleme boşluğunu</span> daraltmaktır. Kelime setinde robust doğrulukların standarttan yüksek çıkması, augmentasyonun bu ortamda düzenlileştirici etki yaptığını gösterir; cümle görevinde ise segment hizalama hataları ana gürültü kaynağı olduğundan skorlar görece düşük ve değişkendir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.5 Cümle düzeyi protokolü ve Test/TestsResults çıktıları
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                        <code className="text-xs opacity-90">sentence_methods_benchmark_results.txt</code> dosyası, duygu başına <span className={strongClass}>15 sentetik cümle</span> ve üç segmentasyon motoru (VAD, VOSK, WhisperX) kombinasyonlarını raporlar. Aşağıda, üretimle hizalı olarak <span className={strongClass}>VOSK</span> kolonundan alınan, her model için <i>standart</i> ve <i>robust</i> ağırlık çiftlerinin <span className={strongClass}>accuracy</span> ve <span className={strongClass}>macro-F1</span> değerleri grafiklenmiştir. <code className="text-xs opacity-90">sentence_hq_methods_benchmark_results.txt</code> ise duygu başına <span className={strongClass}>10 daha uzun / yoğun HQ cümle</span> ile &quot;orta–üst segment mikrofon&quot; benzetimini yansıtır; RF_Robust&apos;un burada öne çıkması, uzun cümlelerde ağaç topluluğunun prosodiyi daha tutarlı özetlediği yorumuna açıktır (Mastermind&apos;da RF ayrıca Sad vetosu için kullanılır).
                    </p>
                    <div className="grid gap-10 lg:grid-cols-1">
                        <div className="w-full min-h-[380px] h-[420px]">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-4 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                Cümle (sentetik 15×4, VOSK): doğruluk (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...sentenceVoskSynthetic15]} margin={{ top: 8, right: 12, left: 8, bottom: 4 }} barCategoryGap="12%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-26} textAnchor="end" height={62} />
                                    <YAxis width={50} domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                                    <RechartsTooltip contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', borderRadius: '12px' }} />
                                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                                    <Bar dataKey="accStd" name="Accuracy standart" fill={isDark ? '#0ea5e9' : '#0284c7'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="accRob" name="Accuracy robust" fill={isDark ? '#22c55e' : '#16a34a'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full min-h-[380px] h-[420px]">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-4 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                Cümle (sentetik 15×4, VOSK): macro-F1 (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...sentenceVoskSynthetic15]} margin={{ top: 8, right: 12, left: 8, bottom: 4 }} barCategoryGap="12%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-26} textAnchor="end" height={62} />
                                    <YAxis width={50} domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                                    <RechartsTooltip contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', borderRadius: '12px' }} />
                                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                                    <Bar dataKey="f1Std" name="Macro-F1 standart" fill={isDark ? '#c084fc' : '#a855f7'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="f1Rob" name="Macro-F1 robust" fill={isDark ? '#f472b6' : '#db2777'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full min-h-[380px] h-[420px]">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-4 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                Cümle (HQ 10×4, VOSK): doğruluk (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...sentenceVoskHq]} margin={{ top: 8, right: 12, left: 8, bottom: 4 }} barCategoryGap="12%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-26} textAnchor="end" height={62} />
                                    <YAxis width={50} domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                                    <RechartsTooltip contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', borderRadius: '12px' }} />
                                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                                    <Bar dataKey="accStd" name="Accuracy standart" fill={isDark ? '#0ea5e9' : '#0284c7'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="accRob" name="Accuracy robust" fill={isDark ? '#22c55e' : '#16a34a'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full min-h-[380px] h-[420px] mb-4">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-4 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                Cümle (HQ 10×4, VOSK): macro-F1 (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...sentenceVoskHq]} margin={{ top: 8, right: 12, left: 8, bottom: 4 }} barCategoryGap="12%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-26} textAnchor="end" height={62} />
                                    <YAxis width={50} domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11 }} />
                                    <RechartsTooltip contentStyle={{ backgroundColor: isDark ? 'rgba(15, 23, 42, 0.95)' : 'white', borderRadius: '12px' }} />
                                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                                    <Bar dataKey="f1Std" name="Macro-F1 standart" fill={isDark ? '#c084fc' : '#a855f7'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="f1Rob" name="Macro-F1 robust" fill={isDark ? '#f472b6' : '#db2777'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <h3 className={`text-2xl font-bold mt-6 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                        4.6 Kelime–cümle performans farkının özeti
                    </h3>
                    <p className="mb-2 pl-4 border-l-4 border-amber-500/50">
                        Kelime doğruluklarının <span className={strongClass}>%90+</span> bandında, cümle doğruluklarının ise çoğunlukla <span className={strongClass}>%20–45</span> (sentetik 15×4) veya <span className={strongClass}>%25–69</span> (HQ) aralığında kalması beklenen bir <i>dağılım kayması</i> ve <i>hata biriktirmesidir</i>: cümle testinde her segment ayrı sınıflandırılır, hizalama hatası özellik vektörünü bozar, ayrıca uzun kayıtta duygular karışabilir. Bu nedenle üretim hattı, yalnızca tek modelin kelime skoruna değil; Bölüm 3&apos;te anlatılan <span className={strongClass}>global–segment harmanı</span> ve <span className={strongClass}>RF tabanlı veto</span> ile kararları stabilize eder.
                    </p>
                </motion.section>

                {/* 5. KELİME BÖLME METOTLARI */}
                <motion.section id="technical-section-5" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaNetworkWired className="text-teal-500 flex-shrink-0" />
                        5. Segmentasyon ve Kelime Bölme Metotları
                    </h2>
                    <p className="mb-6">
                        Konuşma duygu tanımada segmentasyon, <span className={strongClass}>özellik çıkarımının girdi sınırlarını</span> belirlediği için doğrudan sınıflandırma hatasına dönüşebilir. Bu projede üç yol birlikte değerlendirilir: (i) <span className={strongClass}>enerji ve onset tabanlı VAD</span> (<code className="text-sm opacity-90">SentenceProcessor</code>, Librosa), (ii) <span className={strongClass}>VOSK</span> ile çevrimdışı kelime zaman damgaları, (iii) <span className={strongClass}>WhisperX</span> ile transkripsiyon + zorunlu hizalama (forced alignment). Deneysel arayüzde motor seçilebilir; <span className={strongClass}>Mastermind</span> üretim uç noktası varsayılan olarak VOSK kullanır (<code className="text-sm opacity-90">Backend/stt_service.py</code>, <code className="text-sm opacity-90">transcribe(..., engine=&quot;vosk&quot;)</code>).
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.1 Biçimsel tanım ve zaman indeksleme
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-teal-500/50">
                        Örnekleme oranı <span className="font-mono">f<sub>s</sub></span> ile ayrık sinyal <span className="font-mono">s[n]</span>, sürekli zamanda <span className="font-mono">t = n / f<sub>s</sub></span> ile eşlenir. Bir <span className={strongClass}>segmentasyon</span>, aralıkların sonlu bir listesi <span className="font-mono">{'{'}</span><span className="font-mono">[t<sub>i</sub><sup>−</sup>, t<sub>i</sub><sup>+</sup>]</span><span className="font-mono">{'}'}</span><sub>i=1..S</sub> olarak verilir; her aralıkta kesit <span className="font-mono">s<sub>i</sub>[n] = s[n]</span> · <span className="font-mono">𝟙</span><sub>[n<sub>i</sub><sup>−</sup>, n<sub>i</sub><sup>+</sup>]</sub> üzerinden Librosa/OpenSMILE özellikleri hesaplanır. Amaç, <span className="font-mono">S</span> parçanın birleşiminin konuşmayı örtmesi ve mümkün olduğunca <i>kelime veya nefes birimleriyle</i> örtüşmesidir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.2 Enerji tabanlı VAD ve onset (Librosa / SentenceProcessor)
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-teal-500/50">
                        Test hattında &quot;VAD&quot; kolonu, harici bir STT çağrısı olmadan <code className="text-sm opacity-90">SentenceProcessor._advanced_vad_segmentation</code> ile üretilir. İlk aşamada Librosa <span className="font-mono">effects.split</span>, spektral gücün <span className="font-mono">top_db = 25</span> eşiğine göre sessiz bölgeleri bastırarak <i>konuşma adacıkları</i> çıkarır (<span className="font-mono">frame_length = 2048</span>, <span className="font-mono">hop_length = 512</span>). Uzun adacıklar (<span className="font-mono">Δt &gt; 1</span> s) içinde <span className="font-mono">onset_detect</span> ile enerji vuruşları bulunur; minimum bekleme <span className="font-mono">0.3</span> s ve <span className="font-mono">0.25</span> s altı parçalar birleştirilerek tıkırtı ve yarım heceler filtrelenir. Bu yol, literatürdeki klasik <span className={strongClass}>kısa-süre enerji (STE)</span> VAD ailesine yakındır: çerçeve <span className="font-mono">m</span> için <span className="font-mono">E(m) = Σ<sub>k</sub> s<sup>2</sup>[mH + k]</span> ile özetlenen güç üzerinden eşikleme düşünülebilir.
                    </p>
                    <p className={`mb-3 text-sm text-center opacity-75 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                        Şekil — şematik normalize enerji zarfı (ölçüm verisi değil); VAD&apos;in konuşma bölgesini seçme fikri
                    </p>
                    <div className="w-full h-[220px] mb-10 px-2">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={vadEnergyIllustration} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
                                <defs>
                                    <linearGradient id="vadGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={isDark ? '#2dd4bf' : '#0d9488'} stopOpacity={0.55} />
                                        <stop offset="100%" stopColor={isDark ? '#2dd4bf' : '#0d9488'} stopOpacity={0.05} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                                <XAxis dataKey="frame" tick={{ fontSize: 11 }} stroke={isDark ? '#94a3b8' : '#64748b'} label={{ value: 'Çerçeve indeksi m', position: 'insideBottom', offset: -2, fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                                <YAxis domain={[0, 1]} width={44} tick={{ fontSize: 11 }} stroke={isDark ? '#94a3b8' : '#64748b'} tickFormatter={(v) => `${v}`} label={{ value: 'E (norm.)', angle: -90, position: 'insideLeft', fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                                <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                <Area type="monotone" dataKey="energy" stroke={isDark ? '#5eead4' : '#0f766e'} fill="url(#vadGrad)" strokeWidth={2} name="Enerji" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    <h3 className={`text-2xl font-bold mt-6 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.3 VOSK: çevrimdışı STT ve kelime grafiği
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                        <span className={strongClass}>VOSK</span>, Kaldi tabanlı <span className="font-mono">KaldiRecognizer</span> ile Türkçe küçük model (<code className="text-xs opacity-80">vosk-model-small-tr-0.3</code>) üzerinden kelime düzeyinde JSON çıktı üretir; her kelime için <span className="font-mono">start</span> ve <span className="font-mono">end</span> süreleri saniye cinsinden segment sınırlarını tanımlar. Matematiksel olarak bu, konuşma tanıma grafiğinde en iyi yol aramasının (Viterbi benzeri) bir projeksiyonudur; metin içeriği duygu modelinde kullanılmaz, yalnızca <i>zaman kesitleri</i> duyarlıdır. Avantaj: sunucu dışı düşük gecikme ve deterministik çalışma; dezavantaj: zor telaffuz ve gürültüde sınır kayması.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.4 WhisperX: transkripsiyon ve zorunlu hizalama
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                        <span className={strongClass}>WhisperX</span> önce Whisper ASR ile metin üretir, ardından <span className={strongClass}>zorunlu hizalama (forced alignment)</span> modeliyle karakter/kelime düzeyinde zaman çizelgesi çıkarır. Hizalama, genellikle gizli Markov veya sinir hizalayıcılarıyla gözlenen akustik olasılıkların metin dizgisiyle eşlenmesi olarak formüle edilir; kelime <span className="font-mono">w</span> için <span className="font-mono">(τ<sub>start</sub>, τ<sub>end</sub>)</span> çifti, duygu özelliklerinin çıkarılacağı pencereyi daraltır. Gürültülü koşullarda metin hatası segment sınırlarını kaydırabileceğinden, sonuçlar her zaman VOSK ile aynı olmayabilir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.5 Arayüz, test raporu ve üretim tercihi
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                        Ön yüzde segmentasyon motoru seçimi çeviri anahtarlarıyla etiketlenir (ör. <span className={strongClass}>Vosk (Fast/Offline)</span>, <span className={strongClass}>WhisperX (Accurate)</span>, eski yöntem <span className={strongClass}>VAD</span>). <code className="text-xs opacity-80">Test/evaluate_sentence_models.py</code> üç metodu aynı sentetik cümle kümesinde karşılaştırır; çıktı <code className="text-xs opacity-80">sentence_methods_benchmark_results.txt</code> dosyasına yazılır. Mastermind&apos;ın VOSK&apos;a sabitlenmesi, uçtan uca gecikme, çevrimdışı model yolu ve çoklu model deneylerinde elde edilen sistem davranışının birlikte değerlendirilmesiyle uyumludur — tek bir tablo hücresi üstün görünse bile tüm pipeline maliyeti tek başına optimize edilmez.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                        5.6 Ampirik karşılaştırma: segmentasyon motorunun cümle metriklerine etkisi
                    </h3>
                    <p className="mb-6 text-base opacity-90">
                        Aşağıdaki grafikler, <span className={strongClass}>sentetik 15×4 cümle</span> senaryosunda (<code className="text-xs opacity-80">sentence_methods_benchmark_results.txt</code>) üç segmentasyon kolonundan türetilmiştir. Sol grafik yalnızca <span className={strongClass}>CatBoost (standart)</span> ile motor etkisini izole eder; sağ grafik dört tabüler modelin (<span className={strongClass}>CatBoost, XGBoost, LightGBM, GradBoost</span>) <span className={strongClass}>aritmetik ortalamasını</span> gösterir — çoklu model perspektifi, tek bir skorun şansına göre daha kararlıdır.
                    </p>
                    <div className="grid gap-10 lg:grid-cols-2">
                        <div className="w-full min-h-[300px] h-[340px]">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-3 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                CatBoost — doğruluk ve macro-F1 (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...segmentationCatBoostOnly]} margin={{ top: 8, right: 8, left: 4, bottom: 4 }} barCategoryGap="18%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" tick={{ fontSize: 11, fontWeight: 700 }} stroke={isDark ? '#94a3b8' : '#64748b'} />
                                    <YAxis width={46} domain={[0, 45]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                                    <Bar dataKey="acc" name="Accuracy" fill={isDark ? '#14b8a6' : '#0d9488'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="macroF1" name="Macro-F1" fill={isDark ? '#6366f1' : '#4f46e5'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full min-h-[300px] h-[340px]">
                            <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-3 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                4 model ortalaması — doğruluk ve macro-F1 (%)
                            </h4>
                            <ResponsiveContainer width="100%" height="88%">
                                <BarChart data={[...segmentationQuadMean]} margin={{ top: 8, right: 8, left: 4, bottom: 4 }} barCategoryGap="18%">
                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                    <XAxis dataKey="name" tick={{ fontSize: 11, fontWeight: 700 }} stroke={isDark ? '#94a3b8' : '#64748b'} />
                                    <YAxis width={46} domain={[0, 45]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                                    <Bar dataKey="acc" name="Accuracy" fill={isDark ? '#14b8a6' : '#0d9488'} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="macroF1" name="Macro-F1" fill={isDark ? '#6366f1' : '#4f46e5'} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <p className={`mt-4 text-sm text-center max-w-3xl mx-auto opacity-80 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                        Dört model ortalamasında WhisperX, bu özet tabloda hem doğruluk hem macro-F1 açısından biraz önde görünür; VOSK ise üretimde hız ve çevrimdışı kullanım ile dengelenmiştir. Kesin sıralama model ve veri alt kümesine göre değişebilir.
                    </p>
                </motion.section>

                {/* 6. KÜTÜPHANELER VE TEKNİK STACK */}
                <motion.section id="technical-section-6" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaCode className="text-rose-500 flex-shrink-0" />
                        6. Kullanılan Dil, Kütüphane ve Teknik Stack
                    </h2>
                    <p className="mb-8">
                        Bu bölüm, sistemin <span className={strongClass}>yürütme ortamı</span> ile <span className={strongClass}>yazılım bağımlılıklarını</span> katmanlı biçimde özetler. Amaç, tez veya makale ekinde yer alan &quot;yazılım ortamı&quot; bölümünü teknik olarak doğrulanabilir kılmak ve <i>tekrarlanabilirliği</i> (reproducibility) kolaylaştırmaktır. İstemci tarafı <span className={strongClass}>TypeScript</span> ile statik olarak denetlenir; sunucu ve deney betikleri <span className={strongClass}>Python</span> ekosistemindedir.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                        6.1 Ön uç (client) mimarisi
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                        Arayüz <span className={strongClass}>React 19</span> ve <span className={strongClass}>Vite</span> ile derlenir; bileşen stilleri <span className={strongClass}>Tailwind CSS</span> ile tutarlıdır. <span className={strongClass}>React Router</span> çok sayfalı gezintiyi, <span className={strongClass}>Axios</span> HTTP istemcisini, <span className={strongClass}>i18next</span> çok dilliliği yönetir. Mikro-etkileşimler <span className={strongClass}>Framer Motion</span> ile verilir; şekiller <span className={strongClass}>Recharts</span> üzerinden vektörel olarak çizilir. Mikrofon ve oynatma için <span className={strongClass}>Web Audio API</span> ve MediaStream kullanılır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                        6.2 Sunucu API ve iletişim
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                        Üretim servisi <span className={strongClass}>Flask</span> ile sunulur (<code className="text-sm opacity-90">Backend/app.py</code>); <span className={strongClass}>flask-cors</span> tarayıcı–sunucu köken farkını geliştirme ve test için yapılandırır. Ses dosyaları çok parçalı form ile alınır; yanıtlar JSON biçimindedir. Bu, projedeki fiili yürütme yolunu yansıtır (FastAPI değil).
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                        6.3 Makine öğrenimi, ses işleme ve konuşma–metin
                    </h3>
                    <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                        Tabüler modeller <span className={strongClass}>scikit-learn</span> uyumlu eğitim ve <span className={strongClass}>joblib</span> serileştirmesi ile yüklenir; <span className={strongClass}>CatBoost</span>, <span className={strongClass}>XGBoost</span> ve <span className={strongClass}>LightGBM</span> birincil doğruluk taşıyıcılarıdır. Derin ağlar için isteğe bağlı <span className={strongClass}>TensorFlow / Keras</span> yolu kullanılır. Özellik çıkarımı <span className={strongClass}>Librosa</span> ve <span className={strongClass}>OpenSMILE</span> tabanlı ön işleme (<code className="text-xs opacity-80">preprocessing.py</code>) ile yapılır; sayısal çekirdek <span className={strongClass}>NumPy</span> ve <span className={strongClass}>Pandas</span> üzerindedir. Kelime hizalama <span className={strongClass}>Vosk</span> ve <span className={strongClass}>WhisperX</span> (<code className="text-xs opacity-80">stt_service.py</code>) ile sağlanır.
                    </p>

                    <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                        6.4 Deney betikleri ve sonuç arşivi
                    </h3>
                    <p className="mb-6 pl-4 border-l-4 border-rose-500/50">
                        Ölçümler <code className="text-sm opacity-90">Test/</code> altındaki Python betikleriyle üretilir; çıktılar <code className="text-sm opacity-90">Test/TestsResults/</code> içinde metin ve PNG olarak sürüm kontrolüne alınır. Bağımlılık vektörünü kilitlemek için <code className="text-xs opacity-80">Backend/requirements.txt</code> ile birlikte kayıt altına alınması, bağımsız çalışmalarda (replication) önerilen uygulamadır.
                    </p>

                    <div className={`overflow-x-auto rounded-2xl border ${isDark ? 'border-rose-500/25 bg-slate-900/40' : 'border-rose-200 bg-rose-50/50'}`}>
                        <table className="w-full text-left text-sm md:text-base border-collapse">
                            <thead>
                                <tr className={`${isDark ? 'bg-rose-950/50 text-rose-200' : 'bg-rose-100 text-rose-900'}`}>
                                    <th className="p-4 font-black w-[26%]">Katman</th>
                                    <th className="p-4 font-black">Öne çıkan bileşenler</th>
                                </tr>
                            </thead>
                            <tbody className={isDark ? 'text-slate-300' : 'text-slate-700'}>
                                <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                    <td className="p-4 font-bold text-rose-500 align-top">Ön uç</td>
                                    <td className="p-4">React 19, Vite, TypeScript, Tailwind CSS, Framer Motion, Recharts, React Router, Axios, i18next</td>
                                </tr>
                                <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                    <td className="p-4 font-bold text-rose-500 align-top">API</td>
                                    <td className="p-4">Python 3.x, Flask, flask-cors, JSON REST, multipart form-data (ses)</td>
                                </tr>
                                <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                    <td className="p-4 font-bold text-rose-500 align-top">ML / ses</td>
                                    <td className="p-4">scikit-learn, joblib, CatBoost, XGBoost, LightGBM, TensorFlow (isteğe bağlı), Librosa, OpenSMILE, NumPy, Pandas, SoundFile</td>
                                </tr>
                                <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                    <td className="p-4 font-bold text-rose-500 align-top">STT / segment</td>
                                    <td className="p-4">Vosk (KaldiRecognizer), WhisperX (ASR + align), Librosa tabanlı VAD (<code className="text-xs opacity-80">SentenceProcessor</code>)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </motion.section>

                {/* 7. GÜNCEL TEST SONUÇLARI — ETKİLEŞİMLİ ÖZET */}
                <motion.section id="technical-section-7" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                    <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                        <FaMicrophoneAlt className="text-emerald-500 flex-shrink-0" />
                        7. Canlı Performans ve Test Sonuçları (Test Klasörü)
                    </h2>
                    <p className="mb-6">
                        Bu bölüm, <code className="text-sm opacity-90">Test/TestsResults</code> çıktıları ile sayfa içi görselleştirmeler arasında <span className={strongClass}>köprü</span> kurar. Aşağıdaki <span className={strongClass}>etkileşimli konsol</span> üzerinden özet istatistikleri inceleyebilir, grafikleri sekme bazında açabilir ve rapor dosyalarının rollerini tek tek genişletebilirsiniz. Kuramsal ayrıntılar <span className={strongClass}>Bölüm 3–5</span> ile birlikte okunmalıdır.
                    </p>

                    <div className={`rounded-[2rem] border-2 p-4 md:p-6 mb-8 ${isDark ? 'border-emerald-500/30 bg-slate-900/50 shadow-[0_0_40px_-12px_rgba(16,185,129,0.35)]' : 'border-emerald-200 bg-white shadow-lg shadow-emerald-100/80'}`}>
                        <p className={`text-xs font-bold uppercase tracking-widest mb-4 ${isDark ? 'text-emerald-400' : 'text-emerald-700'}`}>
                            Etkileşimli test konsolu
                        </p>
                        <div className="flex flex-wrap gap-2 mb-6">
                            {testHubTabs.map((t) => (
                                <button
                                    key={t.id}
                                    type="button"
                                    onClick={() => setTestHubTab(t.id)}
                                    className={`px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 ${isDark ? 'ring-offset-slate-900' : 'ring-offset-white'} ${
                                        testHubTab === t.id
                                            ? 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white shadow-md scale-[1.02]'
                                            : isDark
                                              ? 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                                              : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                                    }`}
                                >
                                    {t.label}
                                </button>
                            ))}
                        </div>

                        <AnimatePresence mode="wait">
                            <motion.div
                                key={testHubTab}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -6 }}
                                transition={{ duration: 0.2 }}
                            >
                                {testHubTab === 'ozet' && (
                                    <div>
                                        <p className="mb-4 text-base opacity-90">
                                            Kartlara tıklayarak kısa metodoloji notunu açıp kapatabilirsiniz.
                                        </p>
                                        <div className="grid sm:grid-cols-2 gap-4 mb-6">
                                            {[
                                                {
                                                    key: 'mm_acc',
                                                    title: 'Mastermind cümle doğruluğu',
                                                    value: `${(MastermindMetrics.accuracy * 100).toFixed(1)}%`,
                                                    hint: 'Üretim hibrit hattının sentetik cümle doğrulaması; duygu bazlı F1/P/R Bölüm 3.5 grafiğinde.',
                                                    detail:
                                                        'Değer Frontend içindeki realWorldResults (mastermind_benchmark türevi) özetinden okunur ve Bölüm 3.5 ile tutarlıdır. Kesin sayısal döküm için TestsResults altındaki metin dosyasına başvurun.',
                                                },
                                                {
                                                    key: 'mm_f1',
                                                    title: 'Ortalama sınıf F1 (Mastermind)',
                                                    value: `${(mastermindMacroF1Mean * 100).toFixed(1)}%`,
                                                    hint: 'Dört duygunun F1 ortalaması; sınıf dengesizliğinde saf doğruluktan daha bilgilendiricidir.',
                                                    detail:
                                                        'Makro-F1 literatürde tüm sınıflar eşit ağırlıklı iken, burada dört duygunun aritmetik ortalaması olarak yorumlanan özet bir göstergedir.',
                                                },
                                                {
                                                    key: 'word_cv',
                                                    title: 'Kelime CV — standart ortalama',
                                                    value: `${wordAvgStandardAcc.toFixed(1)}%`,
                                                    hint: '8 modelin standart TurEV 5-fold doğruluk ortalaması (benchmark_result ile uyumlu).',
                                                    detail:
                                                        'Her modelin kendi K-fold ortalaması alınır; sonra bu sayfa için modeller arasında aritmetik ortalama raporlanır. Robust eşi Bölüm 4.2 grafiğinde ikinci çubuk olarak yer alır.',
                                                },
                                                {
                                                    key: 'sent_avg',
                                                    title: 'Cümle (VOSK) ort. doğruluk',
                                                    value: `${avgSynthSentenceAcc.toFixed(1)}% / ${avgHqSentenceAcc.toFixed(1)}%`,
                                                    hint: 'Soldaki: sentetik 15×4; sağdaki: HQ 10×4; VOSK + tabüler modeller accStd ortalaması.',
                                                    detail:
                                                        'Her kümede model listesi biraz farklı uzunlukta olsa da ortalama, ilgili dizideki tüm satırların accStd ortalamasıdır; detaylı çubuk grafikleri Bölüm 4.5 ve aşağıdaki Cümle sekmesinde bulunur.',
                                                },
                                            ].map((card) => {
                                                const isOpen = expandedMetric === card.key;
                                                return (
                                                    <button
                                                        key={card.key}
                                                        type="button"
                                                        onClick={() => setExpandedMetric(isOpen ? null : card.key)}
                                                        className={`text-left rounded-2xl p-5 border transition-all hover:scale-[1.01] active:scale-[0.99] ${isDark ? 'border-emerald-500/25 bg-slate-950/60 hover:border-emerald-400/40' : 'border-emerald-200 bg-emerald-50/40 hover:border-emerald-400'} ${isOpen ? 'ring-2 ring-emerald-500/50' : ''}`}
                                                    >
                                                        <p className={`text-xs font-bold uppercase tracking-wide mb-1 ${isDark ? 'text-emerald-400' : 'text-emerald-800'}`}>{card.title}</p>
                                                        <p className={`text-3xl font-black mb-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>{card.value}</p>
                                                        <p className="text-sm opacity-80 leading-relaxed">{card.hint}</p>
                                                        {isOpen && <p className="mt-3 pt-3 border-t text-sm opacity-90 leading-relaxed border-emerald-500/20">{card.detail}</p>}
                                                        <p className={`mt-2 text-xs font-bold ${isDark ? 'text-emerald-500/80' : 'text-emerald-700'}`}>{isOpen ? 'Kapat' : 'Metodoloji notu →'}</p>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                        <p className="text-sm font-bold mb-2 opacity-80">Hızlı gezinme</p>
                                        <div className="flex flex-wrap gap-2">
                                            {[
                                                { id: 'technical-section-3', label: 'Bölüm 3 — Mastermind' },
                                                { id: 'technical-section-4', label: 'Bölüm 4 — Modeller' },
                                                { id: 'technical-section-5', label: 'Bölüm 5 — Segmentasyon' },
                                                { id: 'technical-section-6', label: 'Bölüm 6 — Stack' },
                                            ].map((l) => (
                                                <button
                                                    key={l.id}
                                                    type="button"
                                                    onClick={() => scrollToTechnicalId(l.id)}
                                                    className={`text-xs md:text-sm font-bold px-3 py-2 rounded-lg border ${isDark ? 'border-slate-600 hover:bg-slate-800' : 'border-slate-200 hover:bg-slate-50'}`}
                                                >
                                                    {l.label}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {testHubTab === 'kelime' && (
                                    <div>
                                        <p className="mb-4 text-sm opacity-90">
                                            Veri kaynağı: <code className="text-xs opacity-80">benchmark_result.txt</code> ve <code className="text-xs opacity-80">benchmark_robust_results.txt</code> (Bölüm 4.2 ile aynı sayılar).
                                        </p>
                                        <div className="h-[380px] w-full">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart data={combinedWordBenchmarks} margin={{ top: 12, right: 16, left: 10, bottom: 8 }} barCategoryGap="14%">
                                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                                    <XAxis dataKey="name" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 11, fontWeight: 700 }} interval={0} angle={-24} textAnchor="end" height={64} />
                                                    <YAxis width={52} stroke={isDark ? '#94a3b8' : '#64748b'} domain={[60, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} />
                                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                                                    <Bar dataKey="Standard_Accuracy" name="Standart (%)" fill={isDark ? '#34d399' : '#059669'} radius={[5, 5, 0, 0]} />
                                                    <Bar dataKey="Robust_Accuracy" name="Robust (%)" fill={isDark ? '#6366f1' : '#4f46e5'} radius={[5, 5, 0, 0]} />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                )}

                                {testHubTab === 'cumle' && (
                                    <div>
                                        <div className="flex flex-wrap gap-2 mb-4">
                                            <button
                                                type="button"
                                                onClick={() => setSentenceSuite('synth')}
                                                className={`px-4 py-2 rounded-xl text-sm font-bold transition-all ${sentenceSuite === 'synth' ? 'bg-teal-600 text-white shadow' : isDark ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-700'}`}
                                            >
                                                Sentetik 15×4
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setSentenceSuite('hq')}
                                                className={`px-4 py-2 rounded-xl text-sm font-bold transition-all ${sentenceSuite === 'hq' ? 'bg-teal-600 text-white shadow' : isDark ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-700'}`}
                                            >
                                                HQ 10×4
                                            </button>
                                        </div>
                                        <p className="mb-3 text-sm opacity-85">
                                            VOSK segmentasyonu; standart ve robust cümle doğruluğu. Ham tablo: <code className="text-xs opacity-80">sentence_*_benchmark_results.txt</code>.
                                        </p>
                                        <div className="h-[400px] w-full">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart
                                                    data={cumleChartData}
                                                    margin={{ top: 8, right: 12, left: 8, bottom: 4 }}
                                                    barCategoryGap="12%"
                                                >
                                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                                    <XAxis dataKey="name" tick={{ fontSize: 10, fontWeight: 700 }} stroke={isDark ? '#94a3b8' : '#64748b'} interval={0} angle={-22} textAnchor="end" height={58} />
                                                    <YAxis width={48} domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fontSize: 10 }} />
                                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                                                    <Bar dataKey="accStd" name="Accuracy standart" fill={isDark ? '#2dd4bf' : '#0d9488'} radius={[4, 4, 0, 0]} />
                                                    <Bar dataKey="accRob" name="Accuracy robust" fill={isDark ? '#a78bfa' : '#7c3aed'} radius={[4, 4, 0, 0]} />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                )}

                                {testHubTab === 'segment' && (
                                    <div>
                                        <p className="mb-3 text-sm opacity-90">
                                            Üç segmentasyon motorunun dört tabüler model ortalaması (Bölüm 5.6 ile aynı özet).
                                        </p>
                                        <div className="h-[320px] w-full">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart data={[...segmentationQuadMean]} margin={{ top: 8, right: 12, left: 8, bottom: 4 }} barCategoryGap="20%">
                                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} vertical={false} />
                                                    <XAxis dataKey="name" tick={{ fontSize: 12, fontWeight: 700 }} stroke={isDark ? '#94a3b8' : '#64748b'} />
                                                    <YAxis width={44} domain={[0, 40]} tickFormatter={(v) => `${v}%`} stroke={isDark ? '#94a3b8' : '#64748b'} />
                                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                                                    <Bar dataKey="acc" name="Accuracy" fill={isDark ? '#14b8a6' : '#0d9488'} radius={[5, 5, 0, 0]} />
                                                    <Bar dataKey="macroF1" name="Macro-F1" fill={isDark ? '#818cf8' : '#4f46e5'} radius={[5, 5, 0, 0]} />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => scrollToTechnicalId('technical-section-5')}
                                            className="mt-4 text-sm font-bold text-emerald-600 dark:text-emerald-400 underline-offset-2 hover:underline"
                                        >
                                            Segmentasyon açıklamasına git →
                                        </button>
                                    </div>
                                )}

                                {testHubTab === 'artifacts' && (
                                    <div>
                                        <p className="mb-4 flex items-center gap-2 text-sm opacity-90">
                                            <FaFolderOpen className="text-emerald-500 flex-shrink-0" />
                                            <span>Dosya satırına tıklayarak açıklamayı genişletin. Kök: <code className="text-xs opacity-80">Test/TestsResults/</code></span>
                                        </p>
                                        <ul className="space-y-2">
                                            {TEST_ARTIFACTS.map((art, idx) => {
                                                const open = openArtifactIndex === idx;
                                                return (
                                                    <li key={`${idx}-${art.file}`}>
                                                        <button
                                                            type="button"
                                                            onClick={() => setOpenArtifactIndex(open ? null : idx)}
                                                            className={`w-full text-left rounded-xl border px-4 py-3 flex items-start justify-between gap-3 transition-colors ${isDark ? 'border-slate-700 bg-slate-950/50 hover:bg-slate-900' : 'border-slate-200 bg-slate-50 hover:bg-white'}`}
                                                        >
                                                            <span className="min-w-0">
                                                                <span className={`font-mono text-sm font-bold break-all ${isDark ? 'text-emerald-300' : 'text-emerald-800'}`}>{art.file}</span>
                                                                {open && <p className="mt-2 text-sm opacity-90 leading-relaxed">{art.summary}</p>}
                                                                {open && (
                                                                    <p className="mt-1 text-xs opacity-70">
                                                                        İlgili bölüm: <span className="font-bold">{art.relatesTo}</span>
                                                                    </p>
                                                                )}
                                                            </span>
                                                            <FaChevronDown className={`flex-shrink-0 mt-1 transition-transform duration-200 ${open ? 'rotate-180' : ''} ${isDark ? 'text-slate-500' : 'text-slate-400'}`} />
                                                        </button>
                                                    </li>
                                                );
                                            })}
                                        </ul>
                                    </div>
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    <details className={`rounded-2xl border p-5 mb-6 ${isDark ? 'border-slate-700 bg-slate-900/30' : 'border-slate-200 bg-slate-50'}`}>
                        <summary className="cursor-pointer font-bold text-lg list-none flex items-center justify-between gap-2">
                            <span>Tekrarlanabilirlik notu</span>
                            <span className="text-xs font-mono opacity-70">genişlet</span>
                        </summary>
                        <p className="mt-4 text-base opacity-90 leading-relaxed">
                            Raporları yeniden üretmek için <code className="text-xs opacity-80">Test/</code> altındaki ilgili betiği (ör. <code className="text-xs opacity-80">benchmark_all.py</code>, <code className="text-xs opacity-80">evaluate_sentence_models.py</code>) aynı Python ortamı ve model ağırlık klasörleriyle çalıştırın. Bağımlılıkların sabitlenmesi için <code className="text-xs opacity-80">Backend/requirements.txt</code> ile birlikte kayıt altına alınması, akademik <i>replication</i> standardına uygundur.
                        </p>
                    </details>

                    <p className={`text-center text-sm italic opacity-75 ${isDark ? 'text-slate-500' : 'text-slate-600'}`}>
                        Etkileşimli özet tam metin raporların yerine geçmez; resmî doğrulama için <code className="text-xs">TestsResults/*.txt</code> çıktılarını esas alın.
                    </p>
                </motion.section>

                {/* FOOTER TEXT */}
                <motion.section initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }} className="pt-12">
                     <p className={`text-center font-bold text-xl italic opacity-60 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                        "Makineler bizim ne söylediğimizi duyabilir; ancak bizim asıl amacımız, makinelerin nasıl hissettiğimizi anlamasını sağlamaktır."
                     </p>
                </motion.section>
                </div>
            </div>
        </div>
    );
};

export default TechnicalInfoPage;
