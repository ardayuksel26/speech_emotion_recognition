import { useEffect, useState, useRef, ReactNode } from 'react';
import { useTranslation, Trans } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';
import { motion, useScroll, useSpring } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie,
    Area, AreaChart
} from 'recharts';
import { FaNetworkWired, FaServer, FaMicrophoneAlt, FaBrain, FaDatabase, FaCode, FaFlask } from 'react-icons/fa';


import { MastermindMetrics } from '../data/realWorldResults';
import { vadEnergyIllustration } from '../data/segmentationBenchmark';
import { emotionPerformanceMetrics, confusionMatrix, experimentalModelsData } from '../data/performanceMetrics';

const turEvData = [
    { name: "Angry", value: 487, color: "#ef4444" },
    { name: "Sad", value: 483, color: "#6366f1" },
    { name: "Calm", value: 408, color: "#14b8a6" },
    { name: "Happy", value: 357, color: "#f59e0b" }
];

/* ─── Sub-components ─── */

const SectionCard = ({ children, isDark }: { children: React.ReactNode; isDark: boolean }) => (
    <div
        className="w-full min-w-0 max-w-full backdrop-blur-md overflow-hidden break-words mx-4 md:mx-0"
        style={{
            padding: 'clamp(16px, 5vw, 64px)',   /* Zorunlu CSS padding (iç boşluk) */
            borderRadius: '40px',                /* Zorunlu CSS yuvarlatma */
            background: isDark ? 'rgba(13,21,41,0.65)' : 'rgba(255,255,255,0.45)',
            border: isDark ? '1px solid rgba(255,255,255,0.07)' : '1px solid rgba(203,213,225,0.6)',
            boxShadow: isDark
                ? `0 0 60px rgba(99,102,241,0.07), inset 0 1px 0 rgba(255,255,255,0.05)`
                : `0 8px 40px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.6)`,
            willChange: 'transform, opacity',
        }}
    >
        {children}
    </div>
);

const SectionTitle = ({
    title, iconColor, isDark
}: { num?: string; icon?: React.ReactNode; title: string; iconColor: string; isDark: boolean }) => (
    <div className="flex items-center gap-3 mb-8">
        <div className="w-1 h-8 rounded-full flex-shrink-0" style={{ background: iconColor }} />
        <h2 className={`text-2xl md:text-3xl font-black tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>
            {title}
        </h2>
    </div>
);

const SubTitle = ({ children, color }: { children: React.ReactNode; color: string; isDark?: boolean }) => (
    <h3
        className="text-xl font-bold mb-4 flex items-center gap-2.5"
        style={{ color, marginTop: '3.5rem' }}
    >
        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 inline-block" style={{ background: color }} />
        {children}
    </h3>
);

const Blockquote = ({ children, isDark }: { children: React.ReactNode; color?: string; isDark: boolean }) => (
    <p className={`mb-6 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{children}</p>
);

const TerminalBlock = ({ children, isDark, accentColor }: { children: React.ReactNode; isDark: boolean; accentColor?: string }) => (
    <div className="mb-6 overflow-hidden shadow-lg" style={{ borderRadius: '24px' }}>
        <div
            className="font-mono text-sm md:text-base leading-relaxed overflow-x-auto custom-scrollbar"
            style={{
                padding: '32px', /* Zorunlu Terminal içi boşluk */
                background: isDark ? '#0d1525' : '#1a2540',
                color: accentColor ? `${accentColor}ee` : (isDark ? '#c4d0f0' : '#c4d0f0'),
            }}
        >
            {children}
        </div>
    </div>
);

const StatPill = ({ label, value, color }: { label: string; value: string; color: string }) => (
    <div
        className="flex flex-col items-center justify-center gap-2 rounded-md"
        style={{ padding: '1.25rem 1.75rem', background: `${color}15`, border: `1px solid ${color}30` }}
    >
        <span className="text-4xl md:text-5xl font-black" style={{ color }}>{value}</span>
        <span className="text-xs md:text-sm font-bold uppercase tracking-widest opacity-70">{label}</span>
    </div>
);

const fadeUp: any = {
    initial: { opacity: 0, y: 20 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, margin: '-50px' },
    transition: { duration: 0.5, ease: 'easeOut' as any },
};

/* ─── LazyMount Component ─── */
const LazyMount = ({ children, minHeight = "400px" }: { children: ReactNode, minHeight?: string }) => {
    const [isVisible, setIsVisible] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "800px" } // Load slightly before it comes into view
        );
        
        if (ref.current) observer.observe(ref.current);
        
        return () => observer.disconnect();
    }, []);

    return (
        <div ref={ref} style={{ minHeight: isVisible ? "auto" : minHeight, width: '100%' }}>
            {isVisible ? children : null}
        </div>
    );
};

/* ─── Main Page ─── */
const TechnicalInfoPage = () => {
    const { t, i18n } = useTranslation();
    const isTr = i18n.language === 'tr';
    const { isDark } = useTheme();
    const { scrollYProgress } = useScroll();
    const scaleX = useSpring(scrollYProgress, { stiffness: 100, damping: 30, restDelta: 0.001 });

    const barChartData = MastermindMetrics.metrics.map(d => ({
        name: d.emotion,
        Precision: parseFloat((d.precision * 100).toFixed(1)),
        Recall: parseFloat((d.recall * 100).toFixed(1)),
        F1: parseFloat((d.f1 * 100).toFixed(1)),
    }));

    const [isMounted, setIsMounted] = useState(false);
    useEffect(() => {
        window.scrollTo(0, 0);
        const timer = setTimeout(() => setIsMounted(true), 150);
        return () => clearTimeout(timer);
    }, []);

    const strongClass = isDark ? 'text-white font-bold' : 'text-slate-900 font-bold';

    const tooltipStyle = {
        backgroundColor: isDark ? 'rgba(10,16,35,0.97)' : 'white',
        border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid #e2e8f0',
        borderRadius: '14px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
    };

    return (
        <div className={`min-h-[100vh] pb-32 font-sans transition-colors duration-700 w-full relative`}>

            {/* Progress Bar */}
            <motion.div
                className="fixed top-0 left-0 right-0 h-[3px] z-50 origin-left"
                style={{
                    scaleX,
                    background: 'linear-gradient(90deg, #6366f1, #a855f7, #ec4899)',
                }}
            />

            {/* ── HERO ── */}
            <div className="relative w-full flex flex-col items-center justify-center px-6 overflow-hidden" style={{ paddingTop: '8rem', paddingBottom: '5rem' }}>
                <div
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] rounded-full blur-[140px] pointer-events-none opacity-25"
                    style={{ background: 'radial-gradient(ellipse, #6366f1 0%, #a855f7 50%, transparent 100%)' }}
                />
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
                    className="relative z-10 max-w-4xl text-center flex flex-col items-center gap-6"
                >

                    <h1 className={`text-4xl md:text-6xl font-black tracking-tight leading-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>
                        {t('tech_hero_title_1')}{' '}
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-fuchsia-500">
                            {t('tech_hero_title_2')}
                        </span>
                    </h1>
                    <p className={`text-lg md:text-xl font-light leading-relaxed max-w-3xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                        {t('tech_page_subtitle')}
                    </p>

                    {/* Stat pills */}
                    <div className="flex flex-wrap justify-center gap-4 mt-4">
                        <StatPill label={t('tech_stat_accuracy')} value={`%${(MastermindMetrics.accuracy * 100).toFixed(1)}`} color="#6366f1" />
                        <StatPill label={t('tech_stat_emotions')} value="4" color="#a855f7" />
                        <StatPill label={t('tech_stat_models')} value="10+" color="#ec4899" />
                        <StatPill label={t('tech_stat_dataset')} value="10,410" color="#14b8a6" />
                    </div>
                </motion.div>
            </div>

            {/* ── SECTIONS ── */}
            <div className="relative z-10 w-full flex justify-center px-4 pb-24">
                <div className="w-full max-w-[1200px] text-base md:text-[1.05rem] leading-relaxed" style={{ display: 'flex', flexDirection: 'column', gap: '5rem', minHeight: '100vh' }}>

                    {isMounted && (
                        <>
                            {/* 1. GİRİŞ */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="01" icon={<FaBrain />} title={t('tech_s1_title')} iconColor="#6366f1" isDark={isDark} />
                            <p className="mb-3 font-semibold text-indigo-500 uppercase tracking-wider text-xs">
                                {t('tech_s1_thesis_label')}
                            </p>
                            <p className="mb-6">
                                <Trans i18nKey="tech_s1_p1"><i className={strongClass}>kelime (word) ve cümle (sentence)</i></Trans>
                            </p>
                            <p className="mb-8">
                                <Trans i18nKey="tech_s1_p2"><span className={strongClass}>Master Ensemble Model</span></Trans>
                            </p>
                            <SubTitle color="#818cf8" isDark={isDark}>{t('tech_s1_1_title')}</SubTitle>
                            <Blockquote color="#6366f1" isDark={isDark}>{t('tech_s1_1_intro')}</Blockquote>
                            <ul className="space-y-4 text-base ml-2 mt-4">
                                <li className="flex items-start gap-4">
                                    <span className="mt-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500 flex-shrink-0" />
                                    <span className="flex-1">
                                        {isTr ? (
                                            <>
                                                <span className={strongClass}>Diller Arası Fonetik Farklılıklar (Cross-lingual Bias):</span> Yabancı dilde eğitilmiş bir model, Türkçe'deki kelime vurgularını (stress) yanlış anlayıp hatalı duygu sınıflandırması yapabilir. Bu projede, modele bizzat Türkçe kelime ve hecelerden oluşan kütüphanelerle antrenman yaptırılarak fonetik uçurumlar kapatılmıştır.
                                            </>
                                        ) : (
                                            <>
                                                <span className={strongClass}>Cross-lingual Phonetic Differences (Cross-lingual Bias):</span> A model trained in a foreign language might misinterpret stress and accents in Turkish, leading to incorrect emotion classification. In this project, the phonetic gaps were bridged by training the model directly with libraries composed of Turkish words and syllables.
                                            </>
                                        )}
                                    </span>
                                </li>
                                <li className="flex items-start gap-4">
                                    <span className="mt-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500 flex-shrink-0" />
                                    <span className="flex-1">
                                        {isTr ? (
                                            <>
                                                <span className={strongClass}>Zaman Sınırları ve Ses Boşlukları (Silence boundaries):</span> İnsanlar genellikle duygusal durumlarda duraklar veya nefes alırlar. Silero VAD ve VOSK tabanlı kesme algoritmalarımız bu sessizlik kısımlarını izole ederek makinenin gürültüyü bir duygu olarak tahmin etmesini önler.
                                            </>
                                        ) : (
                                            <>
                                                <span className={strongClass}>Temporal Boundaries and Silence Gaps:</span> Humans often pause or take breaths in emotional states. Our Silero VAD and VOSK-based segmentation algorithms isolate these silent parts to prevent the machine from predicting background noise as an emotion.
                                            </>
                                        )}
                                    </span>
                                </li>
                                <li className="flex items-start gap-4">
                                    <span className="mt-1.5 w-2.5 h-2.5 rounded-full bg-indigo-500 flex-shrink-0" />
                                    <span className="flex-1">
                                        {isTr ? (
                                            <>
                                                <span className={strongClass}>Robust Katmanlarımız:</span> Üst düzey stüdyo frekanslarındaki veri setleri ile eğitilen modeller gerçek dünyada (arka plan gürültüsü vb.) başarısız olur. Buna karşın, V2 algoritmalarının kelime düzeyindeki keskinliği ile Huggingface HuBERT modelinin cümle düzeyindeki geniş bağlam gücünü ağırlıklı (weighted-fusion) olarak harmanlayan sistemimiz, olağanüstü dayanıklı ve kararlı hale (Noise Resilient) gelmiştir.
                                            </>
                                        ) : (
                                            <>
                                                <span className={strongClass}>Our Robust Layers:</span> Models trained on high-level studio frequency datasets fail in the real world. In contrast, our system blends the sharpness of V2 algorithms at the word level with the wide context power of the HuggingFace HuBERT model at the sentence level in a weighted fusion, making it exceptionally resilient and stable against noise.
                                            </>
                                        )}
                                    </span>
                                </li>
                            </ul>
                        </SectionCard>
                    </motion.div>

                    {/* 2. VERİ SETİ */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="02" icon={<FaDatabase />} title={t('tech_s2_title')} iconColor="#d946ef" isDark={isDark} />

                            <SubTitle color="#e879f9" isDark={isDark}>{t('tech_s2_1_title')}</SubTitle>
                            <Blockquote color="#d946ef" isDark={isDark}>
                                <Trans i18nKey="tech_s2_1_p">
                                    <span className={strongClass}>TurEV-DB</span>
                                    <span className="text-red-500 font-bold">Angry</span>
                                    <span className="text-teal-500 font-bold">Calm</span>
                                    <span className="text-amber-500 font-bold">Happy</span>
                                    <span className="text-indigo-500 font-bold">Sad</span>
                                    <span className="font-bold underline text-[1.1rem]">1,735</span>
                                </Trans>
                            </Blockquote>

                            <div className="flex flex-col md:flex-row items-center justify-center my-10 gap-8">
                                <div className={`w-full md:w-1/2 h-[300px] rounded-2xl border ${isDark ? 'border-white/5 bg-white/2' : 'border-slate-100 bg-white/60'}`}>
                                    <LazyMount minHeight="300px">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <PieChart>
                                                <Pie data={turEvData} cx="50%" cy="50%" innerRadius={65} outerRadius={110} paddingAngle={5} dataKey="value">
                                                    {turEvData.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.color} stroke={isDark ? '#0a0f1d' : '#f4f6fb'} strokeWidth={3} />
                                                    ))}
                                                </Pie>
                                                <RechartsTooltip contentStyle={tooltipStyle} formatter={(value) => [`${value} ${t('tech_pie_tooltip_word')}`, t('tech_pie_tooltip_label')]} />
                                                <Legend wrapperStyle={{ fontSize: '14px' }} />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </LazyMount>
                                </div>
                                <div className={`w-full md:w-1/2 rounded-2xl border p-5 ${isDark ? 'border-white/5 bg-white/2' : 'border-slate-100 bg-white/60'}`}>
                                    <h4 className={`text-lg font-bold mb-3 ${isDark ? 'text-white' : 'text-slate-900'}`}>{t('tech_s2_dist_title')}</h4>
                                    <p className="text-sm mb-4 opacity-80">{t('tech_s2_dist_desc')}</p>
                                    <ul className="list-none space-y-2.5 text-sm font-bold">
                                        {[
                                            { label: 'Angry: 487 (28%)', color: '#ef4444' },
                                            { label: 'Sad: 483 (28%)', color: '#6366f1' },
                                            { label: 'Calm: 408 (24%)', color: '#14b8a6' },
                                            { label: 'Happy: 357 (20%)', color: '#f59e0b' },
                                        ].map((item, i) => (
                                            <li key={i} className="px-4 py-2" style={{ color: item.color }}>
                                                {item.label}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>

                            <SubTitle color="#e879f9" isDark={isDark}>{isTr ? "2.2 Gürültü Artırımı (Data_with_noise)" : "2.2 Noise Augmentation (Data_with_noise)"}</SubTitle>
                            <Blockquote color="#d946ef" isDark={isDark}>
                                {isTr
                                  ? <>Modelin gerçek hayat koşullarında dayanıklı olması için TurEV-DB'deki her temiz ses dosyası 5 farklı gürültü türüyle çoğaltılarak <code className="font-mono bg-fuchsia-500/10 px-1 rounded text-sm">Data_with_noise</code> klasörüne kaydedilmiştir. Bu şekilde 1,735 temiz ses 8,675 gürültülü sese dönüşerek toplam eğitim seti <strong className={strongClass}>10,410 ses dosyasına</strong> ulaşmıştır.</>
                                  : <>To make the model robust in real-world conditions, each clean audio file in TurEV-DB was augmented with 5 different noise types and saved to the <code className="font-mono bg-fuchsia-500/10 px-1 rounded text-sm">Data_with_noise</code> folder. This way, 1,735 clean samples became 8,675 noisy samples, bringing the total training set to <strong className={strongClass}>10,410 audio files</strong>.</>
                                }
                            </Blockquote>

                            {/* Noise augmentation table */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8 mt-2">
                                <div className={`p-5 rounded-2xl border ${isDark ? 'border-fuchsia-500/20 bg-fuchsia-500/5' : 'border-fuchsia-200 bg-fuchsia-50/60'}`}>
                                    <p className="font-bold text-sm mb-3" style={{ color: '#d946ef' }}>{isTr ? "Uygulanan 5 Gürültü Türü" : "5 Applied Noise Types"}</p>
                                    <ul className="space-y-1.5 text-sm">
                                        {[
                                            { label: 'background_cafe', desc: isTr ? 'Kafe arka plan sesi' : 'Café background noise' },
                                            { label: 'pink_medium',     desc: isTr ? 'Orta yoğunluklu pembe gürültü' : 'Medium-intensity pink noise' },
                                            { label: 'white_high',     desc: isTr ? 'Yüksek beyaz gürültü' : 'High-intensity white noise' },
                                            { label: 'white_low',      desc: isTr ? 'Düşük beyaz gürültü' : 'Low-intensity white noise' },
                                            { label: 'white_medium',   desc: isTr ? 'Orta beyaz gürültü' : 'Medium white noise' },
                                        ].map((n, i) => (
                                            <li key={i} className="flex items-start gap-2">
                                                <span className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: '#d946ef' }} />
                                                <span><code className="font-mono text-xs">{n.label}</code> — <span className="opacity-70">{n.desc}</span></span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className={`p-5 rounded-2xl border ${isDark ? 'border-fuchsia-500/20 bg-fuchsia-500/5' : 'border-fuchsia-200 bg-fuchsia-50/60'}`}>
                                    <p className="font-bold text-sm mb-3" style={{ color: '#d946ef' }}>{isTr ? "Duygu Başına Toplam Ses" : "Total Samples per Emotion"}</p>
                                    <ul className="space-y-2 text-sm font-bold">
                                        {[
                                            { label: 'Angry',  clean: 487,  noisy: 2435, color: '#ef4444', bg: '#ef444415' },
                                            { label: 'Sad',    clean: 483,  noisy: 2415, color: '#6366f1', bg: '#6366f115' },
                                            { label: 'Calm',   clean: 408,  noisy: 2040, color: '#14b8a6', bg: '#14b8a615' },
                                            { label: 'Happy',  clean: 357,  noisy: 1785, color: '#f59e0b', bg: '#f59e0b15' },
                                        ].map((item, i) => (
                                            <li key={i} className="flex items-center justify-between px-3 py-1.5 rounded-lg" style={{ background: item.bg }}>
                                                <span style={{ color: item.color }}>{item.label}</span>
                                                <span className="opacity-60 text-xs">{item.clean} + {item.noisy} =</span>
                                                <span style={{ color: item.color }}>{item.clean + item.noisy}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>

                            <SubTitle color="#e879f9" isDark={isDark}>{isTr ? "2.3 Sentetik (Yapay) Cümle Jenerasyon Ağı" : "2.3 Synthetic Sentence Generation Engine"}</SubTitle>
                            <Blockquote color="#d946ef" isDark={isDark}>{t('tech_s2_2_p1')}</Blockquote>
                            <Blockquote color="#d946ef" isDark={isDark}>
                                {t('tech_s2_2_p2_prefix')}{' '}
                                <span className="font-bold underline text-[1.05rem]">{t('tech_s2_2_p2_count')}</span>{' '}
                                {t('tech_s2_2_p2_mid')}{' '}
                                <span className="text-red-500 font-bold">20 Angry</span>,{' '}
                                <span className="text-teal-500 font-bold">20 Calm</span>,{' '}
                                <span className="text-amber-500 font-bold">20 Happy</span>,{' '}
                                {t('tech_s2_2_p2_and')}{' '}
                                <span className="text-indigo-500 font-bold">19 Sad</span>.
                            </Blockquote>
                        </SectionCard>
                    </motion.div>

                    {/* 3. MASTER ENSEMBLE MODEL YAPISI */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="03" icon={<FaServer />} title={isTr ? "3. Master Ensemble Model Mimarisi" : "3. Master Ensemble Model Architecture"} iconColor="#06b6d4" isDark={isDark} />
                            <p className="mb-6">
                                {isTr
                                  ? <>Önceki Mastermind mimarimizin yerini alan yeni <strong className={strongClass}>Master Ensemble Model</strong>, üç katmanlı bir yapıya sahiptir: Vosk ile kelime segmentasyonu ve V2 modellerin arayüz zaman çizelgesi için çalışması, SeaBenSea/HuBERT transformatörünün tam cümle analizi ile nihai duygu kararının belirlenmesi ve son olarak kalibrasyon katsayılarıyla dengeleme. Gerçek hayat testlerinde HuBERT (%60.94) V2 modellerinden (%32) çok daha yüksek performans gösterdiği için, nihai karar ağırlıklı olarak HuBERT üzerinden alınmakta; V2 modelleri ise kelime bazlı zaman çizelgesini beslemektedir.</>
                                  : <>Replacing our previous Mastermind architecture, the new <strong className={strongClass}>Master Ensemble Model</strong> operates in three layers: Vosk-based word segmentation with V2 models running for the UI word timeline, SeaBenSea/HuBERT transformer full-sentence analysis for the final emotion decision, and calibration coefficients for balancing. Since HuBERT (60.94%) outperformed V2 models (32%) significantly in real-world tests, the final decision is driven primarily by HuBERT, while V2 models feed the per-word timeline visualization.</>
                                }
                            </p>

                            <SubTitle color="#22d3ee" isDark={isDark}>{isTr ? "Katman 1: Vosk Segmentasyonu ve V2 Modelleri (Kelime Zaman Çizelgesi)" : "Layer 1: Vosk Segmentation and V2 Models (Word Timeline)"}</SubTitle>
                            <Blockquote color="#06b6d4" isDark={isDark}>
                                {isTr
                                  ? "Ses dosyası öncelikle Vosk motoru ile milisaniyelik zaman damgalarına sahip kelimelere bölünür. Her kelime 1582 boyutlu özellik çıkarımından (OpenSMILE IS10) geçirilir ve CatBoost, LightGBM, XGBoost modellerinden oluşan V2 havuzu ile F1-skorlarına dayalı ağırlıklı oylama yapılır. Bu kelime bazlı tahminler arayüzdeki kelime zaman çizelgesini (timeline) besler. Nihai genel duygu kararında ise aşırı uyumlamayı (overfitting) önlemek için kelime bileşeni yerine sabit eşit bir baz değeri (0.25 her duygu sınıfı için) kullanılır."
                                  : "The audio file is first segmented by the Vosk engine into words with millisecond timestamps. Each word undergoes a 1582-dimensional feature extraction (OpenSMILE IS10) and is evaluated with the V2 pool (CatBoost, LightGBM, XGBoost) using F1-weighted voting. These per-word predictions feed the word timeline displayed in the UI. For the final overall emotion decision, a fixed equal baseline (0.25 per emotion class) is used in place of the dynamic word scores — this prevents word-level noise from overfitting the global decision."
                                }
                            </Blockquote>

                            <SubTitle color="#22d3ee" isDark={isDark}>{isTr ? "Katman 2: HuBERT Tam Cümle Analizi (Ana Karar Verici)" : "Layer 2: HuBERT Full Sentence Analysis (Primary Discriminator)"}</SubTitle>
                            <p className="mb-4">
                                {isTr
                                  ? <>SeaBenSea/HuBERT transformatörü kullanılarak sesin genel melodisi, ritmi ve tonlamasından duygu bazlı global skorlar çıkarılır. Modelin çıktısına önce duygu başına kalibrasyon katsayıları (<code className="font-mono bg-cyan-500/10 px-1 rounded text-sm">angry: 0.90, calm: 1.20, happy: 1.20, sad: 0.85</code>) uygulanır, ardından global ölçekleme faktörü (1.7×) ile çarpılarak normalize edilir. Bu katman, nihai duygu kararının birincil belirleyicisidir.</>
                                  : <>Using the SeaBenSea/HuBERT transformer, emotion-specific global scores are extracted from the voice's overall melody, rhythm, and intonation. Per-emotion calibration weights (<code className="font-mono bg-cyan-500/10 px-1 rounded text-sm">angry: 0.90, calm: 1.20, happy: 1.20, sad: 0.85</code>) are applied to the raw outputs, followed by a global scale factor (1.7×), then normalized. This layer is the primary discriminator for the final emotion decision.</>
                                }
                            </p>

                            <SubTitle color="#22d3ee" isDark={isDark}>{isTr ? "Katman 3: Füzyon ve Kalibrasyon" : "Layer 3: Fusion and Calibration"}</SubTitle>
                            <p className="mb-4">
                                {isTr
                                  ? "Sabit kelime bazı ile HuBERT'in normalize edilmiş skorları ağırlıklı olarak birleştirilir, ardından MASTER_CALIBRATION katsayılarıyla son dengeleme yapılır:"
                                  : "The fixed word baseline and HuBERT's normalized scores are combined with learned weights, followed by a final balancing step with MASTER_CALIBRATION coefficients:"
                                }
                            </p>
                            <TerminalBlock isDark={isDark} accentColor="#67e8f9">
                                <p className="mb-2 opacity-60 text-xs">{isTr ? "// Katman 1: Sabit kelime bazı (tüm duygular için eşit)" : "// Layer 1: Fixed word baseline (equal for all emotions)"}</p>
                                <p className="mb-3"><strong>P</strong><sup>(word)</sup><sub>e</sub> = 0.25</p>
                                <p className="mb-2 opacity-60 text-xs">{isTr ? "// Katman 2: HuBERT cümle analizi (duygu ağırlıkları × 1.7 → normalize)" : "// Layer 2: HuBERT sentence analysis (per-emotion weights × 1.7 → normalize)"}</p>
                                <p className="mb-3"><strong>P</strong><sup>(global)</sup><sub>e</sub> = normalize(<strong>HuBERT</strong>(<strong>x</strong><sub>audio</sub>) · w<sub>e</sub> · 1.7)</p>
                                <p className="mb-2 opacity-60 text-xs">{isTr ? "// Katman 3: Füzyon (HuBERT baskın karar verici)" : "// Layer 3: Fusion (HuBERT as dominant decision maker)"}</p>
                                <p className="mb-3"><strong>Score</strong><sub>e</sub> = <strong>P</strong><sup>(word)</sup><sub>e</sub> · 2.2 + <strong>P</strong><sup>(global)</sup><sub>e</sub> · 1.8</p>
                                <p className="mb-2 opacity-60 text-xs">{isTr ? "// Kalibrasyon: MASTER_CALIBRATION × normalize(Score)" : "// Calibration: MASTER_CALIBRATION × normalize(Score)"}</p>
                                <p><strong>Final</strong><sub>e</sub> = normalize(<strong>Score</strong><sub>e</sub> · <strong>cal</strong><sub>e</sub>)  <span className="opacity-50 text-xs ml-2">// cal = &#123;angry:1.25, happy:1.75, sad:0.50, calm:1.40&#125;</span></p>
                            </TerminalBlock>
                            <Blockquote color="#06b6d4" isDark={isDark}>
                                {isTr
                                  ? <>Sabit baz (0.25) kullanımı sayesinde <strong>P</strong><sup>(word)</sup><sub>e</sub> tüm duygular için eşit olduğundan, nihai sıralamayı belirleyen tek değişken HuBERT'in çıktısıdır. Son aşamada <code className="font-mono bg-cyan-500/10 px-1 rounded">MASTER_CALIBRATION</code> katsayıları (Angry: 1.25, Happy: 1.75, Sad: 0.50, Calm: 1.40) uygulanarak en iyi dengeye (%80.94 Accuracy) ulaşılır.</>
                                  : <>Because <strong>P</strong><sup>(word)</sup><sub>e</sub> is equal for all emotions (0.25 fixed), HuBERT's output is the sole variable that determines the final ranking between emotions. In the last step, <code className="font-mono bg-cyan-500/10 px-1 rounded">MASTER_CALIBRATION</code> coefficients (Angry: 1.25, Happy: 1.75, Sad: 0.50, Calm: 1.40) are applied to achieve the best balance (80.94% Accuracy).</>
                                }
                            </Blockquote>

                            {/* Accuracy stat */}
                            <div
                                className="flex flex-col items-center justify-center py-10 my-6 rounded-2xl"
                                style={{
                                    background: isDark ? 'rgba(99,102,241,0.08)' : 'rgba(99,102,241,0.06)',
                                    border: '1px solid rgba(99,102,241,0.2)',
                                }}
                            >
                                <span className="text-xs font-bold uppercase tracking-widest text-emerald-500 mb-3">{isTr ? "Model Doğruluğu (Accuracy)" : "Model Accuracy"}</span>
                                <span
                                    className="text-7xl md:text-8xl font-black tracking-tighter"
                                    style={{
                                        background: 'linear-gradient(135deg, #34d399, #6366f1)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent',
                                        backgroundClip: 'text',
                                    }}
                                >
                                    %{(MastermindMetrics.accuracy * 100).toFixed(2)}
                                </span>
                                <p className={`mt-4 text-sm max-w-2xl mx-auto text-center opacity-75 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                    {isTr
                                      ? "Bu doğruluk oranı, 320 ses dosyasından oluşan özel test seti (our_voices_for_test) üzerinde yapılan testlerde kanıtlanmıştır."
                                      : "This accuracy rate has been proven in tests conducted on a specific test set (our_voices_for_test) consisting of 320 audio files."
                                    }
                                </p>
                            </div>

                            <div className="w-full min-h-[480px] h-[520px] md:h-[560px] py-4 px-2 sm:px-4">
                                <LazyMount minHeight="480px">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={barChartData} margin={{ top: 16, right: 16, left: 8, bottom: 24 }} barCategoryGap="18%">
                                            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#e2e8f0'} vertical={false} />
                                            <XAxis dataKey="name" stroke={isDark ? '#64748b' : '#94a3b8'} tick={{ fontSize: 14, fontWeight: 'bold' }} />
                                            <YAxis width={56} stroke={isDark ? '#64748b' : '#94a3b8'} domain={[0, 100]} tickFormatter={(val) => `${val}%`} tick={{ fontSize: 12, fontWeight: 600 }} tickMargin={10} dx={-2} />
                                            <RechartsTooltip cursor={{ fill: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }} contentStyle={tooltipStyle} itemStyle={{ fontSize: '14px', fontWeight: 'bold' }} />
                                            <Legend wrapperStyle={{ fontSize: '15px', paddingTop: '24px' }} />
                                            <Bar dataKey="F1" name="F1-Score (%)" fill={isDark ? '#a855f7' : '#9333ea'} radius={[6, 6, 0, 0]} />
                                            <Bar dataKey="Precision" name="Precision (%)" fill={isDark ? '#3b82f6' : '#2563eb'} radius={[6, 6, 0, 0]} />
                                            <Bar dataKey="Recall" name="Recall (%)" fill={isDark ? '#ec4899' : '#db2777'} radius={[6, 6, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </LazyMount>
                            </div>
                            <p className={`text-sm text-center max-w-3xl mx-auto mt-2 mb-2 opacity-70 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                {isTr
                                  ? "* Yukarıdaki grafik, Master Ensemble modelinin 4 duygu sınıfı üzerindeki Precision, Recall ve F1 metriklerini gösterir."
                                  : "* The graph above shows the Precision, Recall, and F1 metrics of the Master Ensemble model across 4 emotion classes."
                                }
                            </p>
                        </SectionCard>
                    </motion.div>

                    {/* 4. SINIFLANDIRMA MODELLERİ */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="04" icon={<FaBrain />} title={t('tech_s4_title')} iconColor="#ec4899" isDark={isDark} />
                            <p className="mb-6">{t('tech_s4_intro')}</p>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_1_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_1_p')}</Blockquote>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_1_metrics')}</Blockquote>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_2_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_2_p')}</Blockquote>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_3_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_3_p')}</Blockquote>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_4_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_4_p')}</Blockquote>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_5_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_5_p')}</Blockquote>

                            <SubTitle color="#f472b6" isDark={isDark}>{t('tech_s4_6_title')}</SubTitle>
                            <Blockquote color="#ec4899" isDark={isDark}>{t('tech_s4_6_p')}</Blockquote>
                        </SectionCard>
                    </motion.div>

                    {/* 5. KELİME BÖLME METOTLARI */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="05" icon={<FaNetworkWired />} title={t('tech_s5_title')} iconColor="#14b8a6" isDark={isDark} />
                            <p className="mb-6">{t('tech_s5_intro')}</p>

                            <SubTitle color="#2dd4bf" isDark={isDark}>{t('tech_s5_1_title')}</SubTitle>
                            <Blockquote color="#14b8a6" isDark={isDark}>{t('tech_s5_1_p')}</Blockquote>

                            <SubTitle color="#2dd4bf" isDark={isDark}>{t('tech_s5_2_title')}</SubTitle>
                            <Blockquote color="#14b8a6" isDark={isDark}>{t('tech_s5_2_p')}</Blockquote>
                            <p className={`mb-3 text-sm text-center opacity-60 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                {t('tech_s5_vad_chart_label')}
                            </p>
                            <div className="w-full h-[220px] mb-8">
                                <LazyMount minHeight="220px">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={vadEnergyIllustration} margin={{ top: 8, right: 16, left: 8, bottom: 4 }}>
                                            <defs>
                                                <linearGradient id="vadGrad" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="0%" stopColor={isDark ? '#2dd4bf' : '#0d9488'} stopOpacity={0.55} />
                                                    <stop offset="100%" stopColor={isDark ? '#2dd4bf' : '#0d9488'} stopOpacity={0.05} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#e2e8f0'} />
                                            <XAxis dataKey="frame" tick={{ fontSize: 11 }} stroke={isDark ? '#64748b' : '#94a3b8'} label={{ value: t('tech_vad_chart_x'), position: 'insideBottom', offset: -2, fill: isDark ? '#64748b' : '#94a3b8', fontSize: 11 }} />
                                            <YAxis domain={[0, 1]} width={44} tick={{ fontSize: 11 }} stroke={isDark ? '#64748b' : '#94a3b8'} tickFormatter={(v) => `${v}`} label={{ value: t('tech_vad_chart_y'), angle: -90, position: 'insideLeft', fill: isDark ? '#64748b' : '#94a3b8', fontSize: 11 }} />
                                            <RechartsTooltip contentStyle={{ ...tooltipStyle, borderRadius: '12px' }} />
                                            <Area type="monotone" dataKey="energy" stroke={isDark ? '#5eead4' : '#0f766e'} fill="url(#vadGrad)" strokeWidth={2} name={t('tech_vad_energy')} />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </LazyMount>
                            </div>

                            <SubTitle color="#2dd4bf" isDark={isDark}>{t('tech_s5_3_title')}</SubTitle>
                            <Blockquote color="#14b8a6" isDark={isDark}>{t('tech_s5_3_p')}</Blockquote>

                            <SubTitle color="#2dd4bf" isDark={isDark}>{t('tech_s5_4_title')}</SubTitle>
                            <Blockquote color="#14b8a6" isDark={isDark}>{t('tech_s5_4_p')}</Blockquote>

                            <SubTitle color="#2dd4bf" isDark={isDark}>{t('tech_s5_5_title')}</SubTitle>
                            <Blockquote color="#14b8a6" isDark={isDark}>{t('tech_s5_5_p')}</Blockquote>


                        </SectionCard>
                    </motion.div>

                    {/* 6. KÜTÜPHANELER VE TEKNİK STACK */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="06" icon={<FaCode />} title={t('tech_s6_title')} iconColor="#ef4444" isDark={isDark} />
                            <p className="mb-8">{t('tech_s6_desc')}</p>

                            {[
                                { key: 'tech_s6_1', color: '#f87171' },
                                { key: 'tech_s6_2', color: '#f87171' },
                                { key: 'tech_s6_3', color: '#f87171' },
                                { key: 'tech_s6_4', color: '#f87171' },
                            ].map((s, i) => (
                                <div key={i}>
                                    <SubTitle color="#f87171" isDark={isDark}>{t(`${s.key}_title`)}</SubTitle>
                                    <Blockquote color="#ef4444" isDark={isDark}>{t(`${s.key}_p`)}</Blockquote>
                                </div>
                            ))}

                            {/* Stack Table */}
                            <div
                                className="overflow-x-auto rounded-2xl mt-6"
                                style={{
                                    border: isDark ? '1px solid rgba(239,68,68,0.2)' : '1px solid rgba(239,68,68,0.25)',
                                    background: isDark ? 'rgba(239,68,68,0.04)' : 'rgba(255,241,242,0.7)',
                                }}
                            >
                                <table className="w-full text-left text-sm md:text-base border-collapse">
                                    <thead>
                                        <tr style={{ background: isDark ? 'rgba(239,68,68,0.12)' : 'rgba(239,68,68,0.1)' }}>
                                            <th className="font-black w-[28%] min-w-[160px]" style={{ paddingLeft: '4rem', paddingRight: '1.5rem', paddingTop: '1.25rem', paddingBottom: '1.25rem', color: isDark ? '#fca5a5' : '#991b1b' }}>{t('tech_stack_layer')}</th>
                                            <th className="px-6 py-5 font-black" style={{ color: isDark ? '#fca5a5' : '#991b1b' }}>{t('tech_stack_components')}</th>
                                        </tr>
                                    </thead>
                                    <tbody className={isDark ? 'text-slate-300' : 'text-slate-700'}>
                                        {[
                                            { layer: t('tech_stack_frontend'), content: 'React 19, Vite, TypeScript, Tailwind CSS, Framer Motion, Recharts, React Router, Axios, i18next' },
                                            { layer: t('tech_stack_api'), content: isTr ? 'Python 3.x, Flask, flask-cors, JSON REST, multipart form-data (ses)' : 'Python 3.x, Flask, flask-cors, JSON REST, multipart form-data (audio)' },
                                            { layer: t('tech_stack_ml'), content: isTr ? 'scikit-learn, joblib, CatBoost, XGBoost, LightGBM, TensorFlow (isteğe bağlı), Librosa, OpenSMILE, NumPy, Pandas, SoundFile' : 'scikit-learn, joblib, CatBoost, XGBoost, LightGBM, TensorFlow (optional), Librosa, OpenSMILE, NumPy, Pandas, SoundFile' },
                                            { layer: t('tech_stack_stt'), content: isTr ? 'Vosk (KaldiRecognizer), WhisperX (ASR + align), Librosa tabanlı VAD (SentenceProcessor)' : 'Vosk (KaldiRecognizer), WhisperX (ASR + align), Librosa-based VAD (SentenceProcessor)' },
                                        ].map((row, i) => (
                                            <tr key={i} style={{ borderTop: isDark ? '1px solid rgba(239,68,68,0.1)' : '1px solid rgba(239,68,68,0.15)' }}>
                                                <td className="font-bold align-top" style={{ paddingLeft: '4rem', paddingRight: '1.5rem', paddingTop: '1.25rem', paddingBottom: '1.25rem', color: '#ef4444' }}>{row.layer}</td>
                                                <td className="px-6 py-5 leading-relaxed">
                                                    {row.content}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </SectionCard>
                    </motion.div>

                    {/* 7. TEST SONUÇLARI VE DEĞERLENDİRME */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="07" icon={<FaMicrophoneAlt />} title={isTr ? "7. Test Sonuçları ve Değerlendirme" : "7. Test Results and Evaluation"} iconColor="#10b981" isDark={isDark} />
                            <p className="mb-8">
                                {isTr
                                  ? <>Geliştirdiğimiz konuşma duygu tanıma (SER) sistemini test etmek için iki farklı yaklaşım ve veri seti kullandık: <span className={strongClass}>Sentetik Cümle Testi</span> ve <span className={strongClass}>Gerçek Hayat Testi</span>. Modellerimiz bu aşamalarda tekil olarak test edildi ve nihayetinde Master Ensemble çatısı altında birleştirildi.</>
                                  : <>We used two different approaches and datasets to test our Speech Emotion Recognition (SER) system: <span className={strongClass}>Synthetic Sentence Test</span> and <span className={strongClass}>Real World Test</span>. Our models were tested individually at these stages and finally integrated under the Master Ensemble framework.</>
                                }
                            </p>

                            {/* 7.1 Test Yöntemleri */}
                            <SubTitle color="#34d399" isDark={isDark}>{isTr ? "1. Test Metodolojileri" : "1. Test Methodologies"}</SubTitle>
                            <div className="grid gap-6 md:grid-cols-2 mb-8 mt-4">
                                <div className={`p-6 rounded-2xl border ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-white/60'}`}>
                                    <h4 className={`font-bold mb-3 ${isDark ? 'text-emerald-400' : 'text-emerald-600'}`}>{isTr ? "Sentetik Cümle Testi (sentencevoice_test)" : "Synthetic Sentence Test (sentencevoice_test)"}</h4>
                                    <p className="text-sm">
                                        {isTr
                                          ? "TurEV-DB veri setindeki bağımsız kelimeler (özneler, yüklemler vb.) algoritmik olarak birleştirilerek anlamlı veya anlamsız ancak fonetik olarak tam cümleler oluşturuldu. Bu cümlelerin aralarına gerçekçi duraklar, yapay zeka ile üretilmiş nefes sesleri ve arka plan gürültüleri eklendi."
                                          : "Independent words (subjects, predicates, etc.) in the TurEV-DB dataset were algorithmically combined to create meaningful or nonsense but phonetically complete sentences. Realistic pauses, AI-generated breath sounds, and background noises were added between these sentences."
                                        }
                                    </p>
                                </div>
                                <div className={`p-6 rounded-2xl border ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-white/60'}`}>
                                    <h4 className={`font-bold mb-3 ${isDark ? 'text-emerald-400' : 'text-emerald-600'}`}>{isTr ? "Gerçek Hayat Testi (our_voices_for_test)" : "Real World Test (our_voices_for_test)"}</h4>
                                    <p className="text-sm">
                                        {isTr
                                          ? "Eğitim setinde (TurEV) hiç bulunmayan, farklı kişilerin (örneğin İlhan, Yağız, Yağmur) kendi sesleriyle kaydettiği toplam 320 adet doğal ve kesintisiz (in-the-wild) cümleden oluşur. Modellerin asıl kalibrasyonu ve \"gerçek dünya\" performansı bu veri seti ile ölçülmüştür."
                                          : "It consists of a total of 320 natural and continuous (in-the-wild) sentences recorded by different people (e.g., İlhan, Yağız, Yağmur) whose voices were not in the training set (TurEV). The main calibration and \"real world\" performance of the models were measured using this dataset."
                                        }
                                    </p>
                                </div>
                            </div>

                            {/* 7.2 Tekil Modeller */}
                            <SubTitle color="#34d399" isDark={isDark}>{isTr ? "2. Tekil Model Performansları (Sentetik vs. Gerçek Hayat)" : "2. Individual Model Performance (Synthetic vs. Real World)"}</SubTitle>
                            <Blockquote color="#10b981" isDark={isDark}>
                                {isTr
                                  ? <>Modellerimizi önce <span className="font-mono">sentencevoice_test</span> (sentetik) üzerinde, ardından <span className="font-mono">our_voices_for_test</span> (gerçek) üzerinde test ettik. Sentetik veride %90'lara varan doğruluk oranları, gerçek hayat testlerinde (farklı konuşmacılar ve doğal ortam) %32'lere kadar düştü.</>
                                  : <>We tested our models first on <span className="font-mono">sentencevoice_test</span> (synthetic) and then on <span className="font-mono">our_voices_for_test</span> (real). Accuracy rates up to 90% in synthetic data dropped to 32% in real-life tests (different speakers and natural environments).</>
                                }
                            </Blockquote>
                            <div className="flex flex-col md:flex-row gap-6 mb-8 mt-4">
                                <div className={`flex-1 p-5 rounded-xl border ${isDark ? 'border-amber-500/30 bg-amber-500/10' : 'border-amber-200 bg-amber-50'}`}>
                                    <p className="font-bold mb-2 text-sm text-center">{isTr ? "Sentetik Cümle Testi (sentencevoice_test)" : "Synthetic Sentence Test (sentencevoice_test)"}</p>
                                    <p className="text-xs text-center opacity-70 mb-3">{isTr ? "Sadece V2 Modelleri (Models_2)" : "Only V2 Models (Models_2)"}</p>
                                    <ul className="text-sm space-y-2 opacity-90 list-none flex flex-col items-center justify-center">
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>LightGBM:</span> <strong className="text-emerald-500">%90.94</strong></li>
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>XGBoost:</span> <strong className="text-emerald-500">%90.31</strong></li>
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>CatBoost:</span> <strong className="text-emerald-500">%84.38</strong></li>
                                    </ul>
                                </div>
                                <div className={`flex-1 p-5 rounded-xl border ${isDark ? 'border-red-500/30 bg-red-500/10' : 'border-red-200 bg-red-50'}`}>
                                    <p className="font-bold mb-2 text-sm text-center">{isTr ? "Gerçek Hayat Testi (our_voices_for_test)" : "Real World Test (our_voices_for_test)"}</p>
                                    <p className="text-xs text-center opacity-70 mb-3">{isTr ? "V2 Bireysel Düşüş & HuBERT" : "V2 Individual Drop & HuBERT"}</p>
                                    <ul className="text-sm space-y-2 opacity-90 list-none flex flex-col items-center justify-center">
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>HuBERT (Huggingface):</span> <strong className="text-amber-500">%60.94</strong></li>
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>CatBoost (V2):</span> <strong className="text-red-500">%32.81</strong></li>
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>LightGBM (V2):</span> <strong className="text-red-500">%31.56</strong></li>
                                        <li className="flex items-center gap-1.5"><span>•</span> <span>WavLM (Huggingface):</span> <strong className="text-red-500">%11.56</strong></li>
                                    </ul>
                                </div>
                            </div>
                            <p className="text-sm opacity-70 mb-8 text-center max-w-4xl mx-auto">
                                {isTr
                                  ? "* Görüldüğü üzere, sadece akustik özelliklere (V2) dayalı modeller gerçek hayattaki tonlamalarda yetersiz kalırken, HuBERT %60.94 ile tek başına en iyi bağlamı yakalamıştır. (Wav2Vec2Turkish %0.00 gibi sıfır-atış başarısızlıkları listeye eklenmemiştir.)"
                                  : "* As can be seen, models based solely on acoustic features (V2) fall short in real-world intonation, while HuBERT alone captured the best context at 60.94%. (Zero-shot failures such as Wav2Vec2Turkish 0.00% are excluded from the list.)"
                                }
                            </p>

                            {/* 7.3 Master Ensemble */}
                            <SubTitle color="#34d399" isDark={isDark}>{isTr ? "3. Birlikten Doğan Güç: Master Ensemble Modeli (%80.94)" : "3. Power of Unity: Master Ensemble Model (80.94%)"}</SubTitle>
                            <Blockquote color="#10b981" isDark={isDark}>
                                {isTr
                                  ? <>Tek başlarına %32 gibi düşük doğruluk oranlarına sahip V2 modelleri, kendi içlerinde "Majority Voting" ile birleşip kelime bazlı sağlam bir temel oluşturduğunda ve bu temel HuBERT'in %60'lık genel cümle anlayışıyla ağırlıklı (1.8x HuBERT, 2.2x V2) olarak harmanlandığında sistemin genel doğruluğu <strong>%80.94</strong>'e fırlamıştır. Aşağıdaki grafikler bu nihai başarının sınıf bazlı detaylarını göstermektedir.</>
                                  : <>When the V2 models, which have low accuracy rates like 32% on their own, combine through "Majority Voting" to create a solid word-based foundation, and this foundation is combined weighted (1.8x HuBERT, 2.2x V2) with HuBERT's 60% general sentence comprehension, the overall accuracy of the system jumps to <strong>80.94%</strong>. The graphs below show the class-level details of this ultimate success.</>
                                }
                            </Blockquote>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6 mb-2">
                                <div className={`p-4 rounded-xl border ${isDark ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-emerald-200 bg-emerald-50/50'}`}>
                                    <p className="font-bold text-sm mb-1" style={{ color: '#10b981' }}>{isTr ? "Türkçe Ses Kalibrasyon Matrisi" : "Turkish Vocal Calibration Matrix"}</p>
                                    <p className="text-xs opacity-80 leading-relaxed">
                                        {isTr 
                                            ? "Modelin duygu tahminlerini dengelemek amacıyla şu katsayılar uygulanır: Angry (1.25), Happy (1.75), Sad (0.50), Calm (1.40)." 
                                            : "The following calibration coefficients are applied to balance emotion predictions: Angry (1.25), Happy (1.75), Sad (0.50), Calm (1.40)."}
                                    </p>
                                </div>
                                <div className={`p-4 rounded-xl border ${isDark ? 'border-blue-500/30 bg-blue-500/5' : 'border-blue-200 bg-blue-50/50'}`}>
                                    <p className="font-bold text-sm mb-1" style={{ color: '#3b82f6' }}>{isTr ? "İşleme Hızı" : "Processing Speed"}</p>
                                    <p className="text-xs opacity-80 leading-relaxed">
                                        {isTr 
                                            ? "Uçtan uca (STT, Öznitelik Çıkarımı, ML oylaması ve Master Ensemble) ortalama dosya başına işleme hızı: ~1.4 saniye." 
                                            : "Average end-to-end processing speed per file (STT, Feature Extraction, ML voting, Master Ensemble): ~1.4 seconds."}
                                    </p>
                                </div>
                            </div>

                            <div className="w-full min-h-[400px] h-[440px] py-4 px-1 sm:px-4 rounded-2xl mt-6">
                                <h4 className={`text-center font-bold text-xs sm:text-sm tracking-widest uppercase mb-6 opacity-60 ${isDark ? 'text-white' : 'text-slate-800'}`}>
                                    {isTr ? "Master Ensemble Sınıf Bazlı F1, Precision ve Recall (%)" : "Master Ensemble Class-Based F1, Precision, and Recall (%)"}
                                </h4>
                                <LazyMount minHeight="400px">
                                    <ResponsiveContainer width="100%" height="88%">
                                        <BarChart data={emotionPerformanceMetrics} margin={{ top: 12, right: 18, left: 10, bottom: 8 }} barCategoryGap="18%">
                                            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#e2e8f0'} vertical={false} />
                                            <XAxis dataKey="emotion" stroke={isDark ? '#64748b' : '#94a3b8'} tick={{ fontSize: 13, fontWeight: 700 }} />
                                            <YAxis width={54} stroke={isDark ? '#64748b' : '#94a3b8'} domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fontWeight: 600 }} tickMargin={8} />
                                            <RechartsTooltip cursor={{ fill: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }} contentStyle={tooltipStyle} itemStyle={{ fontSize: '13px', fontWeight: 'bold' }} formatter={(v: unknown) => `${(v as number).toFixed(1)}%`} />
                                            <Legend wrapperStyle={{ fontSize: '13px', paddingTop: '12px' }} />
                                            <Bar dataKey="precision" name="Precision (%)" fill={isDark ? '#3b82f6' : '#2563eb'} radius={[5, 5, 0, 0]} />
                                            <Bar dataKey="recall" name="Recall (%)" fill={isDark ? '#ec4899' : '#db2777'} radius={[5, 5, 0, 0]} />
                                            <Bar dataKey="f1" name="F1-Score (%)" fill={isDark ? '#a855f7' : '#9333ea'} radius={[5, 5, 0, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </LazyMount>
                            </div>

                            {/* 7.4 Confusion Matrix */}
                            <SubTitle color="#34d399" isDark={isDark}>{isTr ? "Karmaşıklık Matrisi (Confusion Matrix)" : "Confusion Matrix"}</SubTitle>
                            <p className="mb-6 opacity-80 text-sm">
                                {isTr
                                  ? "320 ses dosyasından (her sınıftan 80 adet) oluşan testimizde modelimizin hangi duyguları birbiriyle karıştırdığını aşağıdaki tablodan detaylıca inceleyebilirsiniz."
                                  : "In our test consisting of 320 audio files (80 from each class), you can examine in detail from the table below which emotions our model confuses with each other."
                                }
                            </p>
                            <div className="flex justify-center mb-6 overflow-x-auto">
                                <table className="border-collapse text-sm md:text-base">
                                    <thead>
                                        <tr>
                                            <th className={`p-4 text-xs font-bold opacity-50 text-right align-bottom ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                                                {isTr ? "Gerçek ↓" : "True ↓"}
                                            </th>
                                            <th
                                                colSpan={confusionMatrix.labels.length}
                                                className={`p-3 text-center text-xs font-bold uppercase tracking-widest opacity-60 border-b ${isDark ? 'text-slate-300 border-slate-700' : 'text-slate-600 border-slate-300'}`}
                                            >
                                                {isTr ? "Tahmin →" : "Predicted →"}
                                            </th>
                                        </tr>
                                        <tr>
                                            <th className="p-4" />
                                            {confusionMatrix.labels.map((label) => (
                                                <th key={label} className={`p-4 font-black text-center min-w-[90px] ${label === 'Angry' ? 'text-red-500' : label === 'Calm' ? 'text-teal-500' : label === 'Happy' ? 'text-amber-500' : 'text-indigo-500'
                                                    }`}>{isTr ? t(`emotions.${label.toLowerCase()}`) || label : label}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {confusionMatrix.matrix.map((row, ri) => {
                                            const rowSum = row.reduce((a, b) => a + b, 0);
                                            return (
                                                <tr key={ri}>
                                                    <td className={`p-4 font-black text-right ${confusionMatrix.labels[ri] === 'Angry' ? 'text-red-500' : confusionMatrix.labels[ri] === 'Calm' ? 'text-teal-500' : confusionMatrix.labels[ri] === 'Happy' ? 'text-amber-500' : 'text-indigo-500'
                                                        }`}>{isTr ? t(`emotions.${confusionMatrix.labels[ri].toLowerCase()}`) || confusionMatrix.labels[ri] : confusionMatrix.labels[ri]}</td>
                                                    {row.map((val, ci) => {
                                                        const isCorrect = ri === ci;
                                                        const intensity = Math.round((val / rowSum) * 100);
                                                        return (
                                                            <td key={ci} className={`p-4 text-center font-bold rounded-lg border ${isCorrect
                                                                ? isDark ? 'bg-emerald-500/25 border-emerald-500/40 text-emerald-300' : 'bg-emerald-100 border-emerald-300 text-emerald-800'
                                                                : val > 0
                                                                    ? isDark ? 'bg-red-500/10 border-red-500/20 text-red-300' : 'bg-red-50 border-red-200 text-red-700'
                                                                    : isDark ? 'border-slate-800 text-slate-600' : 'border-slate-200 text-slate-400'
                                                                }`}>
                                                                <span className="text-base">{val}</span>
                                                                <span className="block text-xs opacity-50">{intensity}%</span>
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>

                        </SectionCard>
                    </motion.div>

                    {/* 8. DENEYSEL MODELLER */}
                    <motion.div {...fadeUp}>
                        <SectionCard isDark={isDark}>
                            <SectionTitle num="08" icon={<FaFlask />} title={isTr ? "8. Deneysel Modeller ve Analizleri" : "8. Experimental Models and Analysis"} iconColor="#f59e0b" isDark={isDark} />
                            <p className="mb-8">
                                {isTr
                                  ? <>Projenin araştırma sürecinde <span className="font-mono">sentencevoice_test</span> ve <span className="font-mono">Models</span> gibi klasörlerde çeşitli algoritmalar denenmiştir. %30'un üzerinde başarı gösteren modeller, eğitim/test süreçlerindeki karakteristik özellikleriyle birlikte en yüksek doğrudan düşüğe doğru aşağıda listelenmiştir.</>
                                  : <>During the research process of the project, various algorithms were tested in folders such as <span className="font-mono">sentencevoice_test</span> and <span className="font-mono">Models</span>. Models that performed above 30% are listed below from highest to lowest, along with their characteristic features in the training/testing processes.</>
                                }
                            </p>

                            <div className="space-y-12">
                                {experimentalModelsData.map((model, idx) => (
                                    <div key={idx} className={`p-6 rounded-2xl border ${isDark ? 'border-slate-700 bg-slate-800/40' : 'border-slate-200 bg-white/60'}`}>
                                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-4">
                                            <h3 className={`text-xl font-bold flex items-center gap-2 ${isDark ? 'text-amber-400' : 'text-amber-600'}`}>
                                                <span className={`flex items-center justify-center w-8 h-8 rounded-full text-sm ${isDark ? 'bg-amber-500/20' : 'bg-amber-100'}`}>{idx + 1}</span>
                                                {model.name}
                                            </h3>
                                            <div className={`px-4 py-1.5 rounded-full font-black text-sm border ${isDark ? 'border-amber-500/30 bg-amber-500/10 text-amber-300' : 'border-amber-300 bg-amber-50 text-amber-700'}`}>
                                                {isTr ? `Doğruluk: %${model.accuracy}` : `Accuracy: ${model.accuracy}%`}
                                            </div>
                                        </div>
                                        
                                        <p className="text-sm mb-6 opacity-90 leading-relaxed">
                                            {t(model.descriptionKey)}
                                        </p>

                                        <div className="w-full h-[280px]">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <BarChart data={model.metrics} margin={{ top: 10, right: 10, left: 0, bottom: 0 }} barSize={30}>
                                                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1e293b' : '#e2e8f0'} vertical={false} />
                                                    <XAxis dataKey="emotion" stroke={isDark ? '#64748b' : '#94a3b8'} tick={{ fontSize: 12, fontWeight: 600 }} />
                                                    <YAxis width={40} stroke={isDark ? '#64748b' : '#94a3b8'} domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 10 }} />
                                                    <RechartsTooltip cursor={{ fill: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)' }} contentStyle={tooltipStyle} />
                                                    <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
                                                    <Bar dataKey="precision" name="Precision (%)" fill={isDark ? '#3b82f6' : '#2563eb'} radius={[4, 4, 0, 0]} />
                                                    <Bar dataKey="recall" name="Recall (%)" fill={isDark ? '#ec4899' : '#db2777'} radius={[4, 4, 0, 0]} />
                                                    <Bar dataKey="f1" name="F1-Score (%)" fill={isDark ? '#a855f7' : '#9333ea'} radius={[4, 4, 0, 0]} />
                                                </BarChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </SectionCard>
                    </motion.div>

                    {/* FOOTER QUOTE */}
                    <motion.div {...fadeUp}>
                        <div className="flex justify-center pt-8">
                            <p
                                className="text-center font-bold text-xl italic max-w-2xl"
                                style={{ color: isDark ? 'rgba(203,213,225,0.4)' : 'rgba(51,65,85,0.45)' }}
                            >
                                {t('tech_footer_quote')}
                            </p>
                        </div>
                    </motion.div>
                        </>
                    )}

                </div>
            </div>
        </div>
    );
};

export default TechnicalInfoPage;
