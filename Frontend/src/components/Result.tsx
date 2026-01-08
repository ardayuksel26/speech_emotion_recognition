import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AnalysisResult } from '../types';
import ProbabilityChart from './Results/ProbabilityChart';
import { WordTimeline } from './Results/WordTimeline';
import { ExportButton } from './Results/ExportButton';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaArrowLeft, FaPlay, FaPause, FaDownload, FaShareAlt } from 'react-icons/fa';
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
        }
    };

    const handleEnded = () => {
        setIsPlaying(false);
        setProgress(0);
    };

    const getEmotionColor = (emotion: string) => {
        const colors: Record<string, string> = {
            angry: 'from-rose-500 via-red-500 to-red-600',
            happy: 'from-amber-400 via-orange-400 to-orange-500',
            sad: 'from-blue-400 via-indigo-500 to-indigo-600',
            neutral: 'from-gray-400 via-slate-400 to-slate-500',
            calm: 'from-teal-400 via-emerald-400 to-emerald-500',
            fear: 'from-purple-500 via-violet-500 to-violet-600',
            disgust: 'from-green-500 via-lime-500 to-lime-600',
            surprise: 'from-pink-500 via-fuchsia-500 to-fuchsia-600'
        };
        return colors[emotion.toLowerCase()] || 'from-indigo-500 to-purple-600';
    };

    const gradientClass = getEmotionColor(result.dominant_emotion);
    // Darker/Stronger gradient specifically for text visibility
    const textGradientClass = gradientClass.replace('300', '500').replace('400', '600');

    return (
        <div className={clsx(
            "w-full h-full min-h-[80vh] flex flex-col relative font-sans",
            isDark ? "text-white" : "text-slate-900"
        )}>

            {/* 1. TOP NAVIGATION (Floating & Minimal) */}
            <div className="flex items-center justify-between p-4 z-50">
                <button
                    onClick={onBack}
                    className="flex items-center gap-3 group"
                >
                    <div className={clsx(
                        "w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 group-hover:scale-110",
                        isDark ? "bg-white/10 hover:bg-white/20 text-white" : "bg-white hover:bg-slate-50 text-slate-700 shadow-md hover:shadow-lg"
                    )}>
                        <FaArrowLeft className="text-lg" />
                    </div>
                </button>

                <div className="flex gap-3">
                    <ExportButton result={result} />
                </div>
            </div>


            {/* 2. HERO SECTION (Immersive) */}
            <div className="flex-1 flex flex-col items-center justify-center relative w-full max-w-5xl mx-auto py-10">

                {/* Main Visual */}
                <MotionWrapper className="relative z-10 flex flex-col items-center">

                    {/* Animated Aura */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                        <div className={clsx(
                            "w-[300px] h-[300px] md:w-[500px] md:h-[500px] rounded-full blur-[80px] md:blur-[120px] opacity-40 animate-pulse",
                            `bg-gradient-to-tr ${gradientClass}`
                        )} />
                    </div>

                    {/* Emoji Container */}
                    <div className="relative mb-8 md:mb-12">
                        <div className={clsx(
                            "w-48 h-48 md:w-64 md:h-64 rounded-full flex items-center justify-center shadow-2xl relative z-10",
                            "bg-gradient-to-br backdrop-blur-3xl ring-4 ring-white/20 dark:ring-white/5",
                            gradientClass
                        )}>
                            <div className={clsx(
                                "w-[96%] h-[96%] rounded-full flex items-center justify-center shadow-inner",
                                isDark ? "bg-slate-900/30" : "bg-white/30"
                            )}>
                                <EmotionBadge
                                    emotion={result.dominant_emotion}
                                    size="xl"
                                    showLabel={false}
                                    className="!scale-[2.5]"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Typography */}
                    <div className="text-center space-y-4 relative z-10">
                        <h1 className={clsx(
                            "text-6xl md:text-8xl font-black tracking-tight capitalize drop-shadow-sm",
                            `text-transparent bg-clip-text bg-gradient-to-r ${textGradientClass}`
                        )}>
                            {t(result.dominant_emotion.toLowerCase())}
                        </h1>

                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 dark:bg-black/20 backdrop-blur-md border border-white/20 shadow-sm">
                            <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
                            <span className="text-sm font-bold tracking-widest uppercase opacity-80">
                                {t('confidence')}: {(result.confidence * 100).toFixed(0)}%
                            </span>
                        </div>
                    </div>

                </MotionWrapper>

                {/* Floating Player */}
                {audioUrl && (
                    <MotionWrapper delay={0.2} className="w-full max-w-lg mt-12 z-20">
                        <div className={clsx(
                            "flex items-center gap-5 p-2 pr-6 rounded-[2rem] shadow-xl backdrop-blur-2xl transition-all hover:scale-[1.01] group",
                            isDark ? "bg-slate-900/60 border border-white/10" : "bg-white/80 border border-white/60"
                        )}>
                            <button
                                onClick={togglePlay}
                                className={clsx(
                                    "w-14 h-14 rounded-full flex items-center justify-center shadow-lg text-white transition-all transform group-hover:rotate-180",
                                    `bg-gradient-to-r ${gradientClass}`
                                )}
                            >
                                <span className="transform group-hover:-rotate-180 transition-transform">
                                    {isPlaying ? <FaPause className="text-lg" /> : <FaPlay className="text-lg ml-1" />}
                                </span>
                            </button>

                            <div className="flex-1 flex flex-col justify-center gap-2">
                                <div className="flex items-center justify-between px-1">
                                    <span className="text-[10px] font-bold uppercase tracking-widest opacity-50">Recording.wav</span>
                                    <span className="text-[10px] font-bold uppercase tracking-widest opacity-50">{Math.round(progress)}%</span>
                                </div>
                                <div className="h-2 w-full bg-slate-200 dark:bg-slate-700/50 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full bg-gradient-to-r ${gradientClass} transition-all duration-100 ease-linear shadow-[0_0_10px_rgba(0,0,0,0.2)]`}
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    </MotionWrapper>
                )}

            </div>

            {/* 3. INSIGHTS GRID (Glass Cards) */}
            <div className="w-full mt-8 md:mt-0 px-4 md:px-0 grid grid-cols-1 md:grid-cols-2 gap-6 pb-20 max-w-5xl mx-auto">

                <MotionWrapper delay={0.3}>
                    <div className={clsx(
                        "p-8 rounded-[2.5rem] h-full shadow-lg border backdrop-blur-xl transition-all hover:-translate-y-1 hover:shadow-xl",
                        isDark ? "bg-slate-800/40 border-slate-700/50" : "bg-white/60 border-white/80"
                    )}>
                        <h3 className="text-xs font-black uppercase tracking-widest opacity-40 mb-6 flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                            {t('emotion_distribution')}
                        </h3>
                        <ProbabilityChart probabilities={result.emotions} />
                    </div>
                </MotionWrapper>

                {result.word_timestamps && result.word_timestamps.length > 0 && (
                    <MotionWrapper delay={0.4}>
                        <div className={clsx(
                            "p-8 rounded-[2.5rem] h-full shadow-lg border backdrop-blur-xl transition-all hover:-translate-y-1 hover:shadow-xl",
                            isDark ? "bg-slate-800/40 border-slate-700/50" : "bg-white/60 border-white/80"
                        )}>
                            <h3 className="text-xs font-black uppercase tracking-widest opacity-40 mb-6 flex items-center gap-2">
                                <span className="w-1.5 h-1.5 rounded-full bg-purple-500" />
                                {t('timeline_analysis')}
                            </h3>
                            <WordTimeline
                                wordTimestamps={result.word_timestamps}
                                audioDuration={Math.max(...result.word_timestamps.map(w => w.end)) || 10}
                            />
                        </div>
                    </MotionWrapper>
                )}

            </div>

        </div>
    );
};

const MotionWrapper: React.FC<{ children: React.ReactNode; delay?: number; className?: string }> = ({ children, delay = 0, className }) => (
    <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }} // smooth apple-like ease
        className={className}
    >
        {children}
    </motion.div>
);

export default Result;
