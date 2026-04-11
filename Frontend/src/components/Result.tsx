import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AnalysisResult } from '../types';
import ProbabilityChart from './Results/ProbabilityChart';
import { WordTimeline } from './Results/WordTimeline';
import { ExportButton } from './Results/ExportButton';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaArrowLeft, FaPlay, FaPause, FaRobot, FaBrain, FaChartBar } from 'react-icons/fa';
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
        <div className={clsx(
            "w-full min-h-[85vh] flex flex-col font-sans",
            isDark ? "text-white" : "text-slate-900"
        )}>

            {/* TOP NAVIGATION HUD */}
            <div className="flex items-center justify-between mb-8">
                <button
                    onClick={onBack}
                    className={clsx(
                        "flex items-center gap-3 px-5 py-2.5 rounded-2xl backdrop-blur-md border shadow-sm transition-all duration-300 hover:scale-105 group",
                        isDark ? "bg-slate-800/50 border-white/10 hover:bg-slate-700/60" : "bg-white/60 border-slate-200/60 hover:bg-white"
                    )}
                >
                    <FaArrowLeft className="text-sm opacity-70 group-hover:-translate-x-1 transition-transform" />
                    <span className="font-bold text-sm tracking-wide">{t('back')}</span>
                </button>

                <div className="flex items-center gap-4">
                    <div className="hidden md:flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-400 text-xs font-bold tracking-widest uppercase">
                        <FaBrain className="animate-pulse" />
                        Üst Akıl Analiz Raporu
                    </div>
                    <ExportButton result={result} />
                </div>
            </div>

            {/* DASHBOARD GRID */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full max-w-7xl mx-auto flex-1">

                {/* LEFT: DOMINANT EMOTION DISPLAY (COL-SPAN-5) */}
                <MotionWrapper delay={0.1} className="lg:col-span-5 flex flex-col h-full">
                    <div className={clsx(
                        "relative flex-grow flex flex-col items-center justify-center p-10 rounded-[2.5rem] border shadow-2xl backdrop-blur-2xl overflow-hidden",
                        isDark ? "bg-slate-900/50 border-white/10" : "bg-white/40 border-white/80"
                    )}>
                        
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
                                "w-48 h-48 md:w-56 md:h-56 rounded-full flex items-center justify-center shadow-2xl relative",
                                "bg-gradient-to-br backdrop-blur-3xl ring-4 ring-white/20 dark:ring-white/5",
                                emotionColorClass
                            )}>
                                <div className={clsx(
                                    "w-[95%] h-[95%] rounded-full flex items-center justify-center shadow-inner",
                                    isDark ? "bg-slate-900/40" : "bg-white/40"
                                )}>
                                    <EmotionBadge
                                        emotion={result.dominant_emotion}
                                        size="xl"
                                        showLabel={false}
                                        className="!scale-[2.0]"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Title & Confidence */}
                        <div className="relative z-10 text-center space-y-4 w-full">
                            <h1 className={clsx(
                                "text-6xl font-black tracking-tighter capitalize drop-shadow-md",
                                `text-transparent bg-clip-text bg-gradient-to-r ${glowClass}`
                            )}>
                                {t(result.dominant_emotion.toLowerCase())}
                            </h1>

                            <div className="inline-flex items-center gap-3 px-6 py-2.5 rounded-full bg-white/20 dark:bg-black/30 backdrop-blur-md border border-white/20 shadow-inner">
                                <span className={clsx("w-2.5 h-2.5 rounded-full shadow-[0_0_10px_currentColor] animate-pulse", `bg-${glowClass.split(' ')[0].replace('from-', '')}`)} />
                                <span className="text-sm font-black tracking-widest uppercase opacity-90">
                                    {t('confidence')}: {(result.confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>

                        {/* Audio Player Integrated to Left Card */}
                        {audioUrl && (
                            <div className="relative z-20 w-full mt-12 bg-white/20 dark:bg-black/20 rounded-3xl p-4 border border-white/20 dark:border-white/5 shadow-inner flex items-center gap-4">
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
                                        <span>{currentTime}</span>
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
                            <audio ref={audioRef} src={audioUrl} onTimeUpdate={handleTimeUpdate} onEnded={handleEnded} className="hidden" />
                        )}
                    </div>
                </MotionWrapper>

                {/* RIGHT: ANALYTICS DASHBOARD (COL-SPAN-7) */}
                <div className="lg:col-span-7 flex flex-col gap-6 h-full">
                    
                    {/* Veto Information Box (If Applied) */}
                    {result.veto_info?.applied && (
                        <MotionWrapper delay={0.2}>
                            <div className="w-full p-5 rounded-3xl border border-indigo-400/50 shadow-xl bg-gradient-to-r from-indigo-500/20 to-purple-500/10 backdrop-blur-xl flex items-center gap-5">
                                <div className="hidden sm:flex w-14 h-14 rounded-2xl bg-indigo-500 items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.5)] shrink-0">
                                    <FaRobot className="text-white text-2xl" />
                                </div>
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <div className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
                                        <h3 className="text-xs font-black uppercase tracking-widest text-indigo-600 dark:text-indigo-400">Üst Akıl Veto Müdahalesi</h3>
                                    </div>
                                    <p className="text-sm font-medium opacity-80 leading-relaxed">
                                        Gürültü tespiti sebebiyle ana algoritmalar ezildi. Model kararını <strong className="text-indigo-600 dark:text-indigo-300 font-black">Üzgün</strong> olarak işaretledi. (RF Skoru: <span className="font-mono">%{(result.veto_info.rf_score).toFixed(1)}</span>)
                                    </p>
                                </div>
                            </div>
                        </MotionWrapper>
                    )}

                    {/* Chart & Distribution Analysis */}
                    <MotionWrapper delay={0.3} className="flex-1 min-h-[300px]">
                        <div className={clsx(
                            "w-full h-full p-8 rounded-[2.5rem] border shadow-xl backdrop-blur-xl flex flex-col",
                            isDark ? "bg-slate-900/50 border-white/10" : "bg-white/50 border-white/80"
                        )}>
                            <div className="flex justify-between items-center mb-8">
                                <h3 className="text-sm font-black uppercase tracking-widest opacity-60 flex items-center gap-3">
                                    <FaChartBar className="text-lg opacity-80" />
                                    {t('emotion_distribution')}
                                </h3>
                            </div>
                            
                            <div className="flex-1 overflow-y-auto custom-scrollbar">
                                <ProbabilityChart probabilities={result.emotions} />
                            </div>
                        </div>
                    </MotionWrapper>

                    {/* Word Timeline OR Voting Details */}
                    {((result.word_timestamps && result.word_timestamps.length > 0) || (result.model_details && result.model_details.length > 0)) && (
                         <MotionWrapper delay={0.4}>
                            <div className={clsx(
                                "w-full p-8 rounded-[2.5rem] border shadow-xl backdrop-blur-xl",
                                isDark ? "bg-slate-900/50 border-white/10" : "bg-white/50 border-white/80"
                            )}>
                                
                                {result.word_timestamps && result.word_timestamps.length > 0 && (
                                    <>
                                        <h3 className="text-sm font-black uppercase tracking-widest opacity-60 mb-6 flex items-center gap-3">
                                            <span className="w-2 h-2 rounded-full bg-purple-500" />
                                            {t('timeline_analysis')}
                                        </h3>
                                        <WordTimeline
                                            wordTimestamps={result.word_timestamps}
                                            audioDuration={Math.max(...result.word_timestamps.map(w => w.end)) || 10}
                                        />
                                    </>
                                )}

                                {result.model_details && result.model_details.length > 0 && (
                                    <>
                                        <h3 className="text-sm font-black uppercase tracking-widest opacity-60 mb-6 flex items-center gap-3">
                                            <span className="w-2 h-2 rounded-full bg-amber-500" />
                                            {t('voting_details')}
                                        </h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            {result.model_details.map((detail, idx) => {
                                                const emotionColors: Record<string, string> = { angry: '#ef4444', happy: '#f59e0b', sad: '#6366f1', calm: '#10b981' };
                                                const dotColor = emotionColors[detail.prediction] || '#8b5cf6';
                                                
                                                return (
                                                    <div key={idx} className={clsx(
                                                        "flex items-center p-4 rounded-2xl border shadow-sm",
                                                        isDark ? "bg-slate-800/50 border-slate-700/50" : "bg-white/60 border-slate-200/50"
                                                    )}>
                                                        <div className="flex-1 min-w-0">
                                                            <p className="text-xs font-black uppercase tracking-wider truncate mb-1.5 opacity-70">{detail.model}</p>
                                                            <div className="flex items-center gap-2">
                                                                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: dotColor }} />
                                                                <span className="text-sm font-bold capitalize">{t(detail.prediction)}</span>
                                                                <span className="text-xs font-bold opacity-50 ml-auto whitespace-nowrap">{detail.confidence.toFixed(1)}%</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
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
