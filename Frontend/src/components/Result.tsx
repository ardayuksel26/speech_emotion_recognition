import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AnalysisResult } from '../types';
import ProbabilityChart from './Results/ProbabilityChart';
import { WordTimeline } from './Results/WordTimeline';
import { FrequencyChart } from './Results/FrequencyChart';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaArrowLeft, FaPlay, FaPause, FaChartBar, FaRobot, FaMicrophone, FaNetworkWired, FaBrain } from 'react-icons/fa';
import { clsx } from 'clsx';
import EmotionBadge from './Results/EmotionBadge';

interface ResultProps {
    result: AnalysisResult;
    onBack: () => void;
    audioUrl?: string;
}

const Result: React.FC<ResultProps> = ({
    result,
    onBack,
    audioUrl,
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const [isPlaying, setIsPlaying] = useState(false);
    const audioRef = useRef<HTMLAudioElement>(null);
    const [progress, setProgress] = useState(0);
    const [currentTime, setCurrentTime] = useState("0:00");
    const [duration, setDuration] = useState("0:00");

    const togglePlay = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setProgress((audioRef.current.currentTime / audioRef.current.duration) * 100);
            const mins = Math.floor(audioRef.current.currentTime / 60);
            const secs = Math.floor(audioRef.current.currentTime % 60);
            setCurrentTime(`${mins}:${secs.toString().padStart(2, '0')}`);
        }
    };

    const handleEnded = () => {
        setIsPlaying(false);
        setProgress(0);
        setCurrentTime("0:00");
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            const d = audioRef.current.duration;
            const mins = Math.floor(d / 60);
            const secs = Math.floor(d % 60);
            setDuration(`${mins}:${secs.toString().padStart(2, '0')}`);
        }
    };

    const getEmotionColor = (emotion: string) => {
        const colors: Record<string, string> = {
            angry: 'from-rose-500 via-red-500 to-red-600',
            happy: 'from-amber-400 via-orange-400 to-orange-500',
            sad: 'from-blue-400 via-indigo-500 to-indigo-600',
            neutral: 'from-slate-400 via-gray-400 to-gray-500',
            calm: 'from-teal-400 via-emerald-400 to-emerald-500',
            fear: 'from-purple-500 via-violet-500 to-violet-600',
            disgust: 'from-green-500 via-lime-500 to-lime-600',
            surprise: 'from-pink-500 via-fuchsia-500 to-fuchsia-600'
        };
        return colors[emotion.toLowerCase()] || 'from-indigo-500 to-purple-600';
    };

    const emotionColorClass = getEmotionColor(result.dominant_emotion);
    const glowClass = emotionColorClass.replace('300', '500').replace('400', '600');

    return (
        <div
            className={clsx(
                "w-full min-h-[85vh] flex flex-col font-sans px-4 sm:px-8 md:px-12 lg:px-20 pt-8 lg:pt-12 pb-10",
                isDark ? "text-white" : "text-slate-900"
            )}
            style={{ paddingLeft: '12px', paddingRight: '12px', paddingTop: '20px' }}
        >

            {/* TOP NAVIGATION HUD */}
            <div className="flex flex-wrap items-center justify-between gap-6 mb-8 lg:mb-12 w-full max-w-7xl mx-auto" style={{ padding: '8px' }}>
                <button
                    onClick={onBack}
                    className={clsx(
                        "w-12 h-12 flex items-center justify-center rounded-full transition-all duration-300 !bg-transparent group",
                        "hover:scale-110",
                        isDark ? "text-slate-400 hover:text-indigo-400" : "text-slate-500 hover:text-indigo-600"
                    )}
                    title={t('back') || "Back"}
                >
                    <FaArrowLeft className="w-5 h-5 transition-transform group-hover:-translate-x-1" />
                </button>

            </div>

            {/* DASHBOARD GRID */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 w-full max-w-7xl mx-auto flex-1 lg:px-8" style={{ gap: '24px', padding: '16px' }}>

                {/* LEFT: DOMINANT EMOTION DISPLAY (COL-SPAN-5) */}
                <MotionWrapper delay={0.1} className="lg:col-span-5 flex flex-col h-full">
                    <div
                        className={clsx(
                            "relative flex-grow flex flex-col items-center justify-center p-12 lg:p-16 border shadow-2xl backdrop-blur-2xl overflow-hidden rounded-[2rem]",
                            isDark ? "bg-slate-900/50 border-white/10" : "bg-white/40 border-white/80"
                        )}
                        style={{ padding: '24px', borderRadius: '32px' }}
                    >

                        {/* Interactive Aura */}
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className={clsx(
                                "w-full h-full rounded-full blur-[100px] opacity-30 animate-pulse",
                                `bg-gradient-to-tr ${emotionColorClass}`
                            )} />
                        </div>

                        {/* Huge Emoji Badge */}
                        <div className="relative z-10 mb-8">
                            <div className={clsx(
                                "w-48 h-48 md:w-56 md:h-56 rounded-full flex items-center justify-center relative",
                                `bg-gradient-to-br ${emotionColorClass}`
                            )}>
                                <EmotionBadge
                                    emotion={result.dominant_emotion}
                                    size="xl"
                                    showLabel={false}
                                    className="!scale-[2.0]"
                                />
                            </div>
                        </div>

                        {/* Title & Confidence */}
                        <div className="relative z-10 text-center space-y-4 w-full px-6 md:px-12">
                            <h1 className={clsx(
                                "text-6xl font-black tracking-tighter capitalize drop-shadow-md",
                                `text-transparent bg-clip-text bg-gradient-to-r ${glowClass}`
                            )}>
                                {t(result.dominant_emotion.toLowerCase())}
                            </h1>

                            <div className="inline-flex items-center px-6 py-3 rounded-lg bg-white/20 backdrop-blur-md border border-white/20 shadow-inner">
                                <span className="text-sm font-black tracking-widest uppercase opacity-90">
                                    {t('confidence')}: {(result.confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>

                        {/* Audio Player Integrated to Left Card */}
                        {audioUrl && (
                            <div className="relative z-20 w-full mt-16 bg-white/20 dark:bg-black/30 px-12 py-8 border border-white/20 dark:border-white/10 shadow-inner flex items-center gap-6 rounded-[2.5rem]">
                                <button
                                    onClick={togglePlay}
                                    className={clsx(
                                        "w-12 h-12 shrink-0 rounded-full flex items-center justify-center shadow-lg text-white transition-transform hover:scale-110 active:scale-95",
                                        `bg-gradient-to-br ${emotionColorClass}`
                                    )}
                                >
                                    {isPlaying ? <FaPause /> : <FaPlay className="ml-1" />}
                                </button>

                                <div className="flex-1 space-y-2 relative">
                                    <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest opacity-60 px-1">
                                        <span>Recording.wav</span>
                                        <span>{currentTime} / {duration}</span>
                                    </div>
                                    <div className="h-2 w-full bg-slate-300/50 dark:bg-slate-700/50 rounded-full overflow-hidden">
                                        <div
                                            className={clsx(
                                                "h-full transition-all duration-100 ease-linear shadow-[0_0_10px_rgba(255,255,255,0.5)]",
                                                `bg-gradient-to-r ${emotionColorClass}`
                                            )}
                                            style={{ width: `${progress}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}

                        {audioUrl && (
                            <audio ref={audioRef} src={audioUrl} onTimeUpdate={handleTimeUpdate} onEnded={handleEnded} onLoadedMetadata={handleLoadedMetadata} className="hidden" />
                        )}
                    </div>
                </MotionWrapper>

                {/* RIGHT: ANALYTICS DASHBOARD (COL-SPAN-7) */}
                <div className="lg:col-span-7 flex flex-col gap-6 h-full">
                    {/* Chart & Distribution Analysis */}
                    <MotionWrapper delay={0.3} className="flex-1 min-h-[300px]">
                        <div
                            className={clsx(
                                "w-full h-full p-10 lg:p-14 border shadow-xl backdrop-blur-xl flex flex-col rounded-[2rem]",
                                isDark ? "bg-slate-900/50 border-white/10" : "bg-white/50 border-white/80"
                            )}
                            style={{ padding: '24px', borderRadius: '32px' }}
                        >
                            <div className="flex justify-between items-center mb-8 px-4 md:px-6">
                                <h3 className="text-sm font-black uppercase tracking-widest opacity-60 flex items-center gap-3">
                                    <FaChartBar className="text-lg opacity-80" />
                                    {t('emotion_distribution')}
                                </h3>
                            </div>

                            <div className="flex-1">
                                <ProbabilityChart probabilities={result.emotions} />
                            </div>
                        </div>
                    </MotionWrapper>

                    {/* Word Timeline OR Voting Details */}
                    {((result.word_timestamps && result.word_timestamps.length > 0) || result.model_details) && (
                        <MotionWrapper delay={0.4}>
                            <div
                                className={clsx(
                                    "w-full h-full p-10 lg:p-14 border shadow-xl backdrop-blur-xl flex flex-col rounded-[2rem] overflow-hidden",
                                    isDark ? "bg-slate-900/50 border-white/10" : "bg-white/50 border-white/80"
                                )}
                                style={{ padding: '24px', borderRadius: '32px' }}
                            >

                                {result.frequency_data && result.frequency_data.length > 0 ? (
                                    <div className="mb-14 pb-10 border-b border-white/10">
                                        <div className="px-4 md:px-6 mb-4">
                                            <h3 className="text-sm font-black uppercase tracking-widest opacity-60 mb-6 flex items-center gap-3">
                                                <span className="w-2 h-2 rounded-full bg-indigo-500" />
                                                {t('acoustic_report')}
                                            </h3>
                                        </div>
                                        <div className="px-2 md:px-6">
                                            <FrequencyChart 
                                                data={result.frequency_data} 
                                            />
                                        </div>
                                    </div>
                                ) : result.word_timestamps && result.word_timestamps.length > 0 && (
                                    <>
                                        <div className="px-4 md:px-6 mb-4">
                                            <h3 className="text-sm font-black uppercase tracking-widest opacity-60 mb-6 flex items-center gap-3">
                                                <span className="w-2 h-2 rounded-full bg-purple-500" />
                                                {t('timeline_analysis')}
                                            </h3>
                                        </div>
                                        <div className="px-2 md:px-6">
                                            <WordTimeline
                                                wordTimestamps={result.word_timestamps}
                                                audioDuration={Math.max(...result.word_timestamps.map(w => w.end)) || 10}
                                                dominantEmotion={result.dominant_emotion}
                                            />
                                        </div>
                                    </>
                                )}

                                {/* Model Details — Master Ensemble veya Experimental Voting */}
                                {result.model_details && (
                                    Array.isArray(result.model_details) ? (
                                        /* ── Eski format: Experimental Voting Kartları ── */
                                        result.model_details.length > 0 && (
                                            <>
                                                <div className="px-4 md:px-6 mb-4">
                                                    <h3 className="text-sm font-black uppercase tracking-widest opacity-60 mb-6 flex items-center gap-3">
                                                        <span className="w-2 h-2 rounded-full bg-amber-500" />
                                                        {t('voting_details')}
                                                    </h3>
                                                </div>
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 px-2 md:px-6">
                                                    {(result.model_details as Array<{model:string;key:string;weight:number;prediction:string;confidence:number;scores:{[k:string]:number}}>).map((detail, idx) => {
                                                        const emotionColors: Record<string, string> = { angry: '#ef4444', happy: '#f59e0b', sad: '#6366f1', calm: '#10b981' };
                                                        const dotColor = emotionColors[detail.prediction.toLowerCase()] || '#8b5cf6';
                                                        return (
                                                            <div key={idx} className={clsx(
                                                                "flex items-center rounded-2xl border shadow-sm transition-all duration-300",
                                                                isDark ? "bg-slate-800/50 border-slate-700/50" : "bg-white/60 border-slate-200/50"
                                                            )} style={{ padding: '12px', borderRadius: '16px' }}>
                                                                <div className="flex-1 min-w-0">
                                                                    <div className="flex justify-between items-center mb-1">
                                                                        <p className="text-[10px] font-black uppercase tracking-widest opacity-50">{detail.model}</p>
                                                                        <span className="text-[10px] font-bold opacity-30">wt: {detail.weight}x</span>
                                                                    </div>
                                                                    <div className="flex items-center gap-2">
                                                                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: dotColor }} />
                                                                        <span className="text-sm font-bold capitalize">{t(detail.prediction.toLowerCase())}</span>
                                                                        <span className="text-xs font-bold opacity-50 ml-auto whitespace-nowrap">{detail.confidence.toFixed(1)}%</span>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </>
                                        )
                                    ) : (
                                        /* ── Yeni format: Master Ensemble Panel ── */
                                        (() => {
                                            const md = result.model_details as {v2_models:string[];v2_word_count:number;hubert_available:boolean;hubert_emotion:string|null;vosk_error:string|null};
                                            const modelLabels: Record<string, string> = { cb_v2: 'CatBoost V2', lgbm_v2: 'LightGBM V2', xgb_v2: 'XGBoost V2' };
                                            const emotionColors: Record<string, string> = { angry: '#ef4444', happy: '#f59e0b', sad: '#6366f1', calm: '#10b981' };
                                            return (
                                                <div className="px-2 md:px-6 space-y-5">
                                                    <h3 className="text-sm font-black uppercase tracking-widest opacity-60 flex items-center gap-3">
                                                        <FaNetworkWired className="text-indigo-400" />
                                                        Master Ensemble
                                                    </h3>

                                                    {/* V2 Model Chips */}
                                                    <div className="flex flex-wrap gap-2">
                                                        {md.v2_models.map((key) => (
                                                            <span key={key} className={clsx(
                                                                "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border",
                                                                isDark ? "bg-indigo-500/15 border-indigo-500/30 text-indigo-300" : "bg-indigo-50 border-indigo-200 text-indigo-700"
                                                            )}>
                                                                <FaRobot className="text-[10px]" />
                                                                {modelLabels[key] || key}
                                                            </span>
                                                        ))}
                                                    </div>

                                                    {/* Stats Row */}
                                                    <div className="grid grid-cols-2 gap-3">
                                                        {/* Kelime Sayısı */}
                                                        <div className={clsx(
                                                            "flex items-center gap-3 p-4 rounded-2xl border",
                                                            isDark ? "bg-slate-800/60 border-slate-700/50" : "bg-white/60 border-slate-200/50"
                                                        )}>
                                                            <FaMicrophone className={clsx("text-lg shrink-0", isDark ? "text-purple-400" : "text-purple-500")} />
                                                            <div>
                                                                <p className="text-[10px] font-black uppercase tracking-widest opacity-50">Kelime Analizi</p>
                                                                <p className="text-lg font-black">
                                                                    {md.v2_word_count > 0 ? `${md.v2_word_count} kelime` : (md.vosk_error ? 'Vosk Hatası' : 'Kelime yok')}
                                                                </p>
                                                            </div>
                                                        </div>

                                                        {/* HuBERT Durumu */}
                                                        <div className={clsx(
                                                            "flex items-center gap-3 p-4 rounded-2xl border",
                                                            isDark ? "bg-slate-800/60 border-slate-700/50" : "bg-white/60 border-slate-200/50"
                                                        )}>
                                                            <FaBrain className={clsx("text-lg shrink-0", md.hubert_available ? (isDark ? "text-emerald-400" : "text-emerald-600") : "text-slate-400")} />
                                                            <div>
                                                                <p className="text-[10px] font-black uppercase tracking-widest opacity-50">HuBERT Jürisi</p>
                                                                {md.hubert_available && md.hubert_emotion ? (
                                                                    <p className="text-sm font-black flex items-center gap-2">
                                                                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: emotionColors[md.hubert_emotion] || '#8b5cf6' }} />
                                                                        <span className="capitalize">{t(md.hubert_emotion)}</span>
                                                                    </p>
                                                                ) : (
                                                                    <p className="text-sm font-bold opacity-40">Devre Dışı</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {/* Vosk Hatası Uyarısı */}
                                                    {md.vosk_error && (
                                                        <div className="px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400 text-xs font-semibold flex items-center gap-2">
                                                            <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" />
                                                            Vosk segmentasyonu başarısız — sadece HuBERT ile analiz yapıldı
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })()
                                    )
                                )}

                            </div>
                        </MotionWrapper>
                    )}

                </div>
            </div>
        </div>
    );
};

const MotionWrapper: React.FC<{ children: React.ReactNode; delay?: number; className?: string }> = ({ children, delay = 0, className }) => (
    <motion.div
        initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
        animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
        transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }}
        className={className}
    >
        {children}
    </motion.div>
);

export default Result;
