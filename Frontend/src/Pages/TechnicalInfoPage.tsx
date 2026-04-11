import { useEffect, useMemo, useState } from 'react';
import { useTranslation, Trans } from 'react-i18next';
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

const scrollToTechnicalId = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

const TechnicalInfoPage = () => {
    const { t } = useTranslation();
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

    const testArtifacts: TestArtifact[] = useMemo(() => [
        { file: 'benchmark_result.txt', summary: t('tech_art_0_summary'), relatesTo: t('tech_art_0_ref') },
        { file: 'benchmark_robust_results.txt', summary: t('tech_art_1_summary'), relatesTo: t('tech_art_1_ref') },
        { file: 'word_methods_benchmark_results.txt', summary: t('tech_art_2_summary'), relatesTo: t('tech_art_2_ref') },
        { file: 'sentence_methods_benchmark_results.txt', summary: t('tech_art_3_summary'), relatesTo: t('tech_art_3_ref') },
        { file: 'sentence_hq_methods_benchmark_results.txt', summary: t('tech_art_4_summary'), relatesTo: t('tech_art_4_ref') },
        { file: 'mastermind_benchmark_results.txt', summary: t('tech_art_5_summary'), relatesTo: t('tech_art_5_ref') },
        { file: 'mastermind_confusion_matrix_result.txt', summary: t('tech_art_6_summary'), relatesTo: t('tech_art_6_ref') },
        { file: 'validation_robust_metrics.txt', summary: t('tech_art_7_summary'), relatesTo: t('tech_art_7_ref') },
        { file: 'emotion_performance_metrics.txt', summary: t('tech_art_8_summary'), relatesTo: t('tech_art_8_ref') },
        { file: 'benchmark_robust_results.png, benchmark_results.png, …', summary: t('tech_art_9_summary'), relatesTo: t('tech_art_9_ref') },
        // eslint-disable-next-line react-hooks/exhaustive-deps
    ], [t]);

    const testHubTabs: { id: typeof testHubTab; label: string }[] = [
        { id: 'ozet', label: t('tech_tab_summary') },
        { id: 'kelime', label: t('tech_tab_word') },
        { id: 'cumle', label: t('tech_tab_sentence') },
        { id: 'segment', label: t('tech_tab_segment') },
        { id: 'artifacts', label: t('tech_tab_artifacts') },
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
                            {t('tech_hero_title_1')} <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-fuchsia-500">
                                {t('tech_hero_title_2')}
                            </span>
                        </h1>
                        <p className={`text-xl font-light leading-relaxed max-w-4xl mx-auto px-4 ${isDark ? 'opacity-80' : 'opacity-70'}`}>
                            {t('tech_page_subtitle')}
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
                            {t('tech_s1_title')}
                        </h2>
                        <p className="mb-6 font-semibold opacity-90 text-indigo-500 uppercase tracking-wider text-sm">
                            {t('tech_s1_thesis_label')}
                        </p>
                        <p className="mb-6">
                            <Trans i18nKey="tech_s1_p1"><i className={strongClass}>kelime (word) ve cümle (sentence)</i></Trans>
                        </p>
                        <p>
                            <Trans i18nKey="tech_s1_p2"><span className={strongClass}>Mastermind (Üst Akıl)</span></Trans>
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-indigo-300' : 'text-indigo-600'}`}>
                            {t('tech_s1_1_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-indigo-500/50">
                            {t('tech_s1_1_intro')}
                        </p>
                        <ul className="space-y-4 text-base list-disc list-inside ml-4 mb-6">
                            <li><span className={strongClass}>{t('tech_s1_li1_title')}</span> {t('tech_s1_li1')}</li>
                            <li><span className={strongClass}>{t('tech_s1_li2_title')}</span> {t('tech_s1_li2')}</li>
                            <li><Trans i18nKey="tech_s1_li3"><span className={strongClass}>{t('tech_s1_li3_title')}</span><span className={strongClass}>Robust Katmanlarımız</span></Trans></li>
                        </ul>
                    </motion.section>

                    {/* 2. VERİ SETİ */}
                    {/* 2. VERİ SETİ */}
                    <motion.section initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                        <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                            <FaDatabase className="text-fuchsia-500 flex-shrink-0" />
                            {t('tech_s2_title')}
                        </h2>

                        <h3 className={`text-2xl font-bold mt-8 mb-4 ${isDark ? 'text-fuchsia-300' : 'text-fuchsia-600'}`}>
                            {t('tech_s2_1_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                            <Trans i18nKey="tech_s2_1_p"><span className={strongClass}>TurEV-DB</span><span className="text-red-500 font-bold">Angry</span><span className="text-teal-500 font-bold">Calm</span><span className="text-amber-500 font-bold">Happy</span><span className="text-indigo-500 font-bold">Sad</span><span className="font-bold underline text-[1.1rem]">1,735</span></Trans>
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
                                            formatter={(value) => [`${value} ${t('tech_pie_tooltip_word')}`, t('tech_pie_tooltip_label')]}
                                        />
                                        <Legend wrapperStyle={{ fontSize: '15px' }} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="w-full md:w-1/2">
                                <h4 className={`text-xl font-bold mb-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>{t('tech_s2_dist_title')}</h4>
                                <p className="text-base mb-4 opacity-80">{t('tech_s2_dist_desc')}</p>
                                <ul className="space-y-2 font-bold text-sm">
                                    <li className="text-red-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500"></span> Angry: 487 (28%)</li>
                                    <li className="text-indigo-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-indigo-500"></span> Sad: 483 (28%)</li>
                                    <li className="text-teal-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-teal-500"></span> Calm: 408 (24%)</li>
                                    <li className="text-amber-500 flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-amber-500"></span> Happy: 357 (20%)</li>
                                </ul>
                            </div>
                        </div>

                        <h3 className={`text-2xl font-bold mt-8 mb-4 ${isDark ? 'text-fuchsia-300' : 'text-fuchsia-600'}`}>
                            {t('tech_s2_2_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                            {t('tech_s2_2_p1')}
                        </p>
                        <p className="mb-6 pl-4 border-l-4 border-fuchsia-500/50">
                            <Trans i18nKey="tech_s2_2_p2"><span className="font-bold underline text-[1.1rem]">79 Pürüzsüz Cümle</span><span className="text-red-500 font-bold">20 Angry</span><span className="text-teal-500 font-bold">20 Calm</span><span className="text-amber-500 font-bold">20 Happy</span><span className="text-indigo-500 font-bold">19 Sad</span></Trans>
                        </p>
                    </motion.section>

                    {/* 3. MASTERMIND YAPISI */}
                    <motion.section id="technical-section-3" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                        <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                            <FaServer className="text-cyan-500 flex-shrink-0" />
                            {t('tech_s3_title')}
                        </h2>
                        <p className="mb-6">
                            {t('tech_s3_intro')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                            {t('tech_s3_1_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-cyan-500/50">
                            {t('tech_s3_1_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                            {t('tech_s3_2_title')}
                        </h3>
                        <p className="mb-4">
                            {t('tech_s3_2_p')}
                        </p>
                        <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base leading-relaxed overflow-x-auto ${isDark ? 'bg-slate-900/80 border border-cyan-500/20 text-cyan-100/95' : 'bg-slate-50 border border-cyan-200/80 text-slate-800'}`}>
                            <p className="mb-3"><span className="opacity-70">Global probability vector:</span> <strong>P</strong><sup>(g)</sup> = softmax / <strong>predict_proba</strong>(<strong>x</strong><sub>global</sub>)</p>
                            <p className="mb-3"><span className="opacity-70">Average over S segments:</span> <strong>P</strong><sup>(seg)</sup> = (1/S) Σ<sub>s=1..S</sub> <strong>P</strong>(<strong>x</strong><sub>s</sub>)</p>
                            <p className="mb-0"><span className="opacity-70">Blended distribution (backend constant):</span> <strong>p̂</strong><sup>(m)</sup> = 0.60 · <strong>P</strong><sup>(g)</sup> + 0.40 · <strong>P</strong><sup>(seg)</sup></p>
                        </div>
                        <p className="mb-6 pl-4 border-l-4 border-cyan-500/50">
                            {t('tech_s3_2_blend')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                            {t('tech_s3_3_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-cyan-500/50">
                            {t('tech_s3_3_p')}
                        </p>
                        <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base ${isDark ? 'bg-slate-900/80 border border-purple-500/20 text-purple-100/95' : 'bg-indigo-50 border border-indigo-200/80 text-slate-800'}`}>
                            main<sub>k</sub> = ( score<sub>k</sub><sup>(CatBoost)</sup> + score<sub>k</sub><sup>(XGBoost)</sup> ) / 2
                        </div>
                        <p className="mb-6">
                            {t('tech_s3_3_decision')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                            {t('tech_s3_4_title')}
                        </h3>
                        <p className="mb-4">
                            {t('tech_s3_4_p')}
                        </p>
                        <ul className="space-y-3 mb-6 text-base list-disc list-inside ml-2">
                            <li><span className={strongClass}>{t('tech_s3_4_rule1_title')}</span> {t('tech_s3_4_rule1')}</li>
                            <li><span className={strongClass}>{t('tech_s3_4_rule2_title')}</span> {t('tech_s3_4_rule2')}</li>
                        </ul>
                        <p className="mb-6 pl-4 border-l-4 border-cyan-500/50 opacity-95">
                            {t('tech_s3_4_summary')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-cyan-300' : 'text-cyan-600'}`}>
                            {t('tech_s3_5_title')}
                        </h3>
                        <p className="mb-6">
                            {t('tech_s3_5_p')}
                        </p>

                        <div className="flex flex-col items-center justify-center mb-10 mt-2 text-center">
                            <span className="text-sm md:text-base font-bold uppercase tracking-widest text-emerald-500 mb-2 opacity-90">
                                {t('tech_mm_acc_label')}
                            </span>
                            <span className={`text-6xl md:text-7xl font-black tracking-tighter drop-shadow-lg ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                %{(MastermindMetrics.accuracy * 100).toFixed(1)}
                            </span>
                            <p className={`mt-4 text-sm md:text-base max-w-2xl mx-auto opacity-80 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                {t('tech_mm_f1_label')}
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
                            {t('tech_s3_5_note')}
                        </p>
                    </motion.section>

                    {/* 4. SINIFLANDIRMA MODELLERİ */}
                    <motion.section id="technical-section-4" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                        <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                            <FaFlask className="text-amber-500 flex-shrink-0" />
                            {t('tech_s4_title')}
                        </h2>
                        <p className="mb-8">
                            {t('tech_s4_intro')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                            {t('tech_s4_1_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_1_p')}
                        </p>
                        <div className={`mb-6 p-5 rounded-2xl font-mono text-sm md:text-base leading-relaxed overflow-x-auto ${isDark ? 'bg-slate-900/80 border border-amber-500/20 text-amber-100/95' : 'bg-amber-50/90 border border-amber-200/80 text-slate-800'}`}>
                            <p className="mb-2">Class <span className="font-mono">k</span>: Precision<sub>k</sub> = TP<sub>k</sub> / (TP<sub>k</sub> + FP<sub>k</sub>), Recall<sub>k</sub> = TP<sub>k</sub> / (TP<sub>k</sub> + FN<sub>k</sub>)</p>
                            <p className="mb-0">F1<sub>k</sub> = 2 · Precision<sub>k</sub> · Recall<sub>k</sub> / (Precision<sub>k</sub> + Recall<sub>k</sub>) &nbsp;·&nbsp; Macro-F1 = (1/C) Σ<sub>k</sub> F1<sub>k</sub></p>
                        </div>
                        <p className="mb-6 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_1_metrics')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                            {t('tech_s4_2_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_2_p')}
                        </p>
                        <p className="mb-6 text-base opacity-90">
                            {t('tech_s4_2_chart_desc')}
                        </p>
                        <div className="w-full min-h-[400px] h-[440px] py-4 px-1 sm:px-4 rounded-[2rem]">
                            <h4 className={`text-center font-bold text-xs sm:text-sm tracking-widest uppercase mb-6 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                {t('tech_chart_word_title')}
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
                            {t('tech_s4_3_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_3_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                            {t('tech_s4_4_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_4_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-amber-300' : 'text-amber-600'}`}>
                            {t('tech_s4_5_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_5_p')}
                        </p>
                        <div className="grid gap-10 lg:grid-cols-1">
                            <div className="w-full min-h-[380px] h-[420px]">
                                <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-4 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                    {t('tech_chart_sent_synth_acc')}
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
                                    {t('tech_chart_sent_synth_f1')}
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
                                    {t('tech_chart_sent_hq_acc')}
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
                                    {t('tech_chart_sent_hq_f1')}
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
                            {t('tech_s4_6_title')}
                        </h3>
                        <p className="mb-2 pl-4 border-l-4 border-amber-500/50">
                            {t('tech_s4_6_p')}
                        </p>
                    </motion.section>

                    {/* 5. KELİME BÖLME METOTLARI */}
                    <motion.section id="technical-section-5" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                        <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                            <FaNetworkWired className="text-teal-500 flex-shrink-0" />
                            {t('tech_s5_title')}
                        </h2>
                        <p className="mb-6">
                            {t('tech_s5_intro')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_1_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-teal-500/50">
                            {t('tech_s5_1_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_2_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-teal-500/50">
                            {t('tech_s5_2_p')}
                        </p>
                        <p className={`mb-3 text-sm text-center opacity-75 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            {t('tech_s5_vad_chart_label')}
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
                                    <XAxis dataKey="frame" tick={{ fontSize: 11 }} stroke={isDark ? '#94a3b8' : '#64748b'} label={{ value: t('tech_vad_chart_x'), position: 'insideBottom', offset: -2, fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                                    <YAxis domain={[0, 1]} width={44} tick={{ fontSize: 11 }} stroke={isDark ? '#94a3b8' : '#64748b'} tickFormatter={(v) => `${v}`} label={{ value: t('tech_vad_chart_y'), angle: -90, position: 'insideLeft', fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                                    <RechartsTooltip contentStyle={{ borderRadius: '12px', backgroundColor: isDark ? 'rgba(15,23,42,0.95)' : '#fff' }} />
                                    <Area type="monotone" dataKey="energy" stroke={isDark ? '#5eead4' : '#0f766e'} fill="url(#vadGrad)" strokeWidth={2} name={t('tech_vad_energy')} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        <h3 className={`text-2xl font-bold mt-6 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_3_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                            {t('tech_s5_3_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_4_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                            {t('tech_s5_4_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_5_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-teal-500/50">
                            {t('tech_s5_5_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-teal-300' : 'text-teal-600'}`}>
                            {t('tech_s5_6_title')}
                        </h3>
                        <p className="mb-6 text-base opacity-90">
                            {t('tech_s5_6_p')}
                        </p>
                        <div className="grid gap-10 lg:grid-cols-2">
                            <div className="w-full min-h-[300px] h-[340px]">
                                <h4 className={`text-center font-bold text-xs tracking-widest uppercase mb-3 opacity-75 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                    {t('tech_chart_catboost')}
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
                                    {t('tech_chart_quad')}
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
                            {t('tech_seg_note')}
                        </p>
                    </motion.section>

                    {/* 6. KÜTÜPHANELER VE TEKNİK STACK */}
                    <motion.section id="technical-section-6" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }}>
                        <h2 className={`text-3xl font-black mb-8 flex items-center gap-4 border-b-2 pb-4 ${isDark ? 'text-white border-slate-700' : 'text-slate-900 border-slate-200'}`}>
                            <FaCode className="text-rose-500 flex-shrink-0" />
                            {t('tech_s6_title')}
                        </h2>
                        <p className="mb-8">
                            {t('tech_s6_desc')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                            {t('tech_s6_1_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                            {t('tech_s6_1_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                            {t('tech_s6_2_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                            {t('tech_s6_2_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                            {t('tech_s6_3_title')}
                        </h3>
                        <p className="mb-4 pl-4 border-l-4 border-rose-500/50">
                            {t('tech_s6_3_p')}
                        </p>

                        <h3 className={`text-2xl font-bold mt-10 mb-4 ${isDark ? 'text-rose-300' : 'text-rose-600'}`}>
                            {t('tech_s6_4_title')}
                        </h3>
                        <p className="mb-6 pl-4 border-l-4 border-rose-500/50">
                            {t('tech_s6_4_p')}
                        </p>

                        <div className={`overflow-x-auto rounded-2xl border ${isDark ? 'border-rose-500/25 bg-slate-900/40' : 'border-rose-200 bg-rose-50/50'}`}>
                            <table className="w-full text-left text-sm md:text-base border-collapse">
                                <thead>
                                    <tr className={`${isDark ? 'bg-rose-950/50 text-rose-200' : 'bg-rose-100 text-rose-900'}`}>
                                        <th className="p-4 font-black w-[26%]">{t('tech_stack_layer')}</th>
                                        <th className="p-4 font-black">{t('tech_stack_components')}</th>
                                    </tr>
                                </thead>
                                <tbody className={isDark ? 'text-slate-300' : 'text-slate-700'}>
                                    <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                        <td className="p-4 font-bold text-rose-500 align-top">{t('tech_stack_frontend')}</td>
                                        <td className="p-4">React 19, Vite, TypeScript, Tailwind CSS, Framer Motion, Recharts, React Router, Axios, i18next</td>
                                    </tr>
                                    <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                        <td className="p-4 font-bold text-rose-500 align-top">{t('tech_stack_api')}</td>
                                        <td className="p-4">Python 3.x, Flask, flask-cors, JSON REST, multipart form-data (ses)</td>
                                    </tr>
                                    <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                        <td className="p-4 font-bold text-rose-500 align-top">{t('tech_stack_ml')}</td>
                                        <td className="p-4">scikit-learn, joblib, CatBoost, XGBoost, LightGBM, TensorFlow (isteğe bağlı), Librosa, OpenSMILE, NumPy, Pandas, SoundFile</td>
                                    </tr>
                                    <tr className={`border-t ${isDark ? 'border-rose-500/15' : 'border-rose-200'}`}>
                                        <td className="p-4 font-bold text-rose-500 align-top">{t('tech_stack_stt')}</td>
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
                            {t('tech_s7_title')}
                        </h2>
                        <p className="mb-6">
                            {t('tech_s7_intro')}
                        </p>

                        <div className={`rounded-[2rem] border-2 p-4 md:p-6 mb-8 ${isDark ? 'border-emerald-500/30 bg-slate-900/50 shadow-[0_0_40px_-12px_rgba(16,185,129,0.35)]' : 'border-emerald-200 bg-white shadow-lg shadow-emerald-100/80'}`}>
                            <p className={`text-xs font-bold uppercase tracking-widest mb-4 ${isDark ? 'text-emerald-400' : 'text-emerald-700'}`}>
                                {t('tech_interactive_console')}
                            </p>
                            <div className="flex flex-wrap gap-2 mb-6">
                                {testHubTabs.map((t) => (
                                    <button
                                        key={t.id}
                                        type="button"
                                        onClick={() => setTestHubTab(t.id)}
                                        className={`px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 ${isDark ? 'ring-offset-slate-900' : 'ring-offset-white'} ${testHubTab === t.id
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
                                                {t('tech_card_click_hint')}
                                            </p>
                                            <div className="grid sm:grid-cols-2 gap-4 mb-6">
                                                {[
                                                    {
                                                        key: 'mm_acc',
                                                        title: t('tech_card_mm_acc_title'),
                                                        value: `${(MastermindMetrics.accuracy * 100).toFixed(1)}%`,
                                                        hint: t('tech_card_mm_acc_hint'),
                                                        detail: t('tech_card_mm_acc_detail'),
                                                    },
                                                    {
                                                        key: 'mm_f1',
                                                        title: t('tech_card_mm_f1_title'),
                                                        value: `${(mastermindMacroF1Mean * 100).toFixed(1)}%`,
                                                        hint: t('tech_card_mm_f1_hint'),
                                                        detail: t('tech_card_mm_f1_detail'),
                                                    },
                                                    {
                                                        key: 'word_cv',
                                                        title: t('tech_card_word_cv_title'),
                                                        value: `${wordAvgStandardAcc.toFixed(1)}%`,
                                                        hint: t('tech_card_word_cv_hint'),
                                                        detail: t('tech_card_word_cv_detail'),
                                                    },
                                                    {
                                                        key: 'sent_avg',
                                                        title: t('tech_card_sent_title'),
                                                        value: `${avgSynthSentenceAcc.toFixed(1)}% / ${avgHqSentenceAcc.toFixed(1)}%`,
                                                        hint: t('tech_card_sent_hint'),
                                                        detail: t('tech_card_sent_detail'),
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
                                                            <p className={`mt-2 text-xs font-bold ${isDark ? 'text-emerald-500/80' : 'text-emerald-700'}`}>{isOpen ? t('tech_close') : t('tech_method_note')}</p>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                            <p className="text-sm font-bold mb-2 opacity-80">{t('tech_quick_nav')}</p>
                                            <div className="flex flex-wrap gap-2">
                                                {[
                                                    { id: 'technical-section-3', label: t('tech_goto_s3') },
                                                    { id: 'technical-section-4', label: t('tech_goto_s4') },
                                                    { id: 'technical-section-5', label: t('tech_goto_s5') },
                                                    { id: 'technical-section-6', label: t('tech_goto_s6') },
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
                                                {t('tech_word_tab_desc')}
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
                                                    {t('tech_synth_btn')}
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setSentenceSuite('hq')}
                                                    className={`px-4 py-2 rounded-xl text-sm font-bold transition-all ${sentenceSuite === 'hq' ? 'bg-teal-600 text-white shadow' : isDark ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-700'}`}
                                                >
                                                    {t('tech_hq_btn')}
                                                </button>
                                            </div>
                                            <p className="mb-3 text-sm opacity-85">
                                                {t('tech_sent_tab_desc')}
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
                                                {t('tech_seg_tab_desc')}
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
                                                {t('tech_goto_seg')}
                                            </button>
                                        </div>
                                    )}

                                    {testHubTab === 'artifacts' && (
                                        <div>
                                            <p className="mb-4 flex items-center gap-2 text-sm opacity-90">
                                                <FaFolderOpen className="text-emerald-500 flex-shrink-0" />
                                                <span>{t('tech_artifacts_hint')}</span>
                                            </p>
                                            <ul className="space-y-2">
                                                {testArtifacts.map((art, idx) => {
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
                                                                            {t('tech_related_section')} <span className="font-bold">{art.relatesTo}</span>
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
                                <span>{t('tech_reproducibility_title')}</span>
                                <span className="text-xs font-mono opacity-70">{t('tech_reproducibility_expand')}</span>
                            </summary>
                            <p className="mt-4 text-base opacity-90 leading-relaxed">
                                {t('tech_reproducibility_text')}
                            </p>
                        </details>

                        <p className={`text-center text-sm italic opacity-75 ${isDark ? 'text-slate-500' : 'text-slate-600'}`}>
                            {t('tech_disclaimer')}
                        </p>
                    </motion.section>

                    {/* FOOTER TEXT */}
                    <motion.section initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true, margin: "-100px" }} className="pt-12">
                        <p className={`text-center font-bold text-xl italic opacity-60 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            {t('tech_footer_quote')}
                        </p>
                    </motion.section>
                </div>
            </div>
        </div>
    );
};

export default TechnicalInfoPage;
