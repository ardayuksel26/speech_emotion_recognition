import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AnalysisResult } from '../types';
import ProbabilityChart from './Results/ProbabilityChart';
import { WordTimeline } from './Results/WordTimeline';
import { FrequencyChart } from './Results/FrequencyChart';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaArrowLeft, FaPlay, FaPause, FaChartBar } from 'react-icons/fa';
import { clsx } from 'clsx';
import EmotionBadge from './Results/EmotionBadge';

interface ResultProps {
    result: AnalysisResult;
    onBack: () => void;
    audioUrl?: string;
    hideTimeline?: boolean;
}

const Result: React.FC<ResultProps> = ({
    result,
    onBack,
    audioUrl,
    hideTimeline = false,
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
                "w-full flex flex-col font-sans",
                isDark ? "text-white" : "text-slate-900"
            )}
            style={{ paddingLeft: '8px', paddingRight: '8px', paddingTop: '0', paddingBottom: '4px' }}
        >

            {/* TOP NAVIGATION HUD */}
            <div className="flex flex-wrap items-center justify-between gap-6 w-full max-w-7xl mx-auto" style={{ padding: '0 4px', marginBottom: '4px' }}>
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
            <div className="grid grid-cols-1 lg:grid-cols-12 w-full max-w-7xl mx-auto" style={{ gap: '16px', padding: '0 4px' }}>

                {/* LEFT: DOMINANT EMOTION DISPLAY (COL-SPAN-4) */}
                <MotionWrapper delay={0.1} className="lg:col-span-4 flex flex-col h-full">
                    <div
                        className={clsx(
                            "relative flex-grow flex flex-col items-center justify-center border shadow-2xl backdrop-blur-2xl overflow-hidden rounded-[2rem]",
                            isDark ? "bg-slate-900/50 border-white/10" : "bg-white/40 border-white/80"
                        )}
                        style={{ padding: '20px', borderRadius: '32px' }}
                    >

                        {/* Interactive Aura */}
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className={clsx(
                                "w-full h-full rounded-full blur-[100px] opacity-30 animate-pulse",
                                `bg-gradient-to-tr ${emotionColorClass}`
                            )} />
                        </div>

                        {/* Emoji Badge */}
                        <div className="relative z-10" style={{ marginBottom: '20px' }}>
                            <div className={clsx(
                                "w-36 h-36 md:w-44 md:h-44 rounded-full flex items-center justify-center relative",
                                `bg-gradient-to-br ${emotionColorClass}`
                            )}>
                                <EmotionBadge
                                    emotion={result.dominant_emotion}
                                    size="xl"
                                    showLabel={false}
                                    className="!scale-[1.6]"
                                />
                            </div>
                        </div>

                        {/* Title & Confidence */}
                        <div className="relative z-10 text-center w-full" style={{ padding: '0 32px' }}>
                            <h1
                                className={clsx(
                                    "text-4xl md:text-5xl font-black tracking-tighter capitalize drop-shadow-md",
                                    `text-transparent bg-clip-text bg-gradient-to-r ${glowClass}`
                                )}
                                style={{ padding: '4px 2px 6px' }}
                            >
                                {t(result.dominant_emotion.toLowerCase())}
                            </h1>

                            <p className="text-sm font-black tracking-widest uppercase opacity-70" style={{ marginTop: '16px' }}>
                                {t('confidence')}: {(result.confidence * 100).toFixed(1)}%
                            </p>
                        </div>

                        {/* Audio Player — pushed to bottom */}
                        {audioUrl && (
                            <div
                                    className="relative z-20 w-full mt-auto pt-6 px-4 py-4 shadow-inner flex items-center gap-4 rounded-[1.5rem]"
                                    style={{
                                        background: isDark ? 'rgba(0,0,0,0.3)' : '#ffffff',
                                        border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(203,213,225,0.4)',
                                        marginTop: '24px',
                                    }}
                                >
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

                {/* RIGHT: ANALYTICS DASHBOARD (COL-SPAN-8) */}
                <div className="lg:col-span-8 flex flex-col gap-6 h-full">
                    {/* Chart & Distribution Analysis */}
                    <MotionWrapper delay={0.3} className={hideTimeline ? '' : 'flex-1 min-h-[300px]'}>
                        <div
                            className={clsx(
                                "w-full border shadow-xl backdrop-blur-xl flex flex-col rounded-[2rem]",
                                isDark ? "bg-slate-900/50 border-white/10" : "bg-white/50 border-white/80"
                            )}
                            style={{
                                padding: hideTimeline ? '16px' : '24px',
                                borderRadius: '32px',
                                minHeight: hideTimeline ? '0' : undefined,
                            }}
                        >
                            <div className={clsx("flex justify-between items-center px-4 md:px-6", hideTimeline ? 'mb-4' : 'mb-8')}>
                                <h3 className="text-sm font-black uppercase tracking-widest opacity-60 flex items-center gap-3">
                                    <FaChartBar className="text-lg opacity-80" />
                                    {t('emotion_distribution')}
                                </h3>
                            </div>

                            <div className={hideTimeline ? '' : 'flex-1'}>
                                <ProbabilityChart probabilities={result.emotions} />
                            </div>
                        </div>
                    </MotionWrapper>

                    {/* Word Timeline OR Voting Details */}
                    {!hideTimeline && ((result.word_timestamps && result.word_timestamps.length > 0) || result.model_details) && (
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
                                {result.model_details && result.frequency_data && result.frequency_data.length > 0 && (
                                    <div style={{ marginTop: '40px' }} />
                                )}
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
                                    ) : null
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
