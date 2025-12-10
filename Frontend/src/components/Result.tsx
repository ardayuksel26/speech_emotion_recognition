import React from 'react';
import { useTranslation } from 'react-i18next';
import { AnalysisResult } from '../types';
import EmotionBadge from './Results/EmotionBadge';
import ConfidenceMeter from './Results/ConfidenceMeter';
import ProbabilityChart from './Results/ProbabilityChart';
import { WordTimeline } from './Results/WordTimeline';
import { ExportButton } from './Results/ExportButton';
import { useTheme } from '../context/ThemeContext';
import { motion } from 'framer-motion';
import { FaChevronLeft, FaShareAlt } from 'react-icons/fa';
import { clsx } from 'clsx';

interface ResultProps {
    result: AnalysisResult;
    onBack: () => void;
    audioUrl?: string;
    waveformLevels?: number[];
}

const Result: React.FC<ResultProps> = ({
    result,
    onBack,
    audioUrl,
    waveformLevels
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    const normalizedDominant = result.dominant_emotion.toLowerCase();
    const dominantProbability = result.emotions[result.dominant_emotion] || result.emotions[normalizedDominant] || 0;

    return (
        <div className={clsx(
            "w-full h-full flex flex-col overflow-y-auto custom-scrollbar",
            isDark ? "text-white" : "text-gray-800"
        )}>
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-100 dark:border-gray-700">
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-sm font-medium opacity-70 hover:opacity-100 transition-opacity"
                >
                    <FaChevronLeft />
                    {t('back')}
                </button>
                <h2 className="text-xl font-bold">{t('analysis_result')}</h2>
                <div className="flex gap-2">
                    <ExportButton result={result} />
                    <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                        <FaShareAlt className="text-gray-500 dark:text-gray-400" />
                    </button>
                </div>
            </div>

            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Left Column: Dominant Emotion & Meter */}
                <div className="flex flex-col items-center justify-center space-y-8 animate-fade-in-up">
                    <MotionWrapper delay={0.1}>
                        <ConfidenceMeter
                            value={dominantProbability}
                            emotion={result.dominant_emotion}
                            size={200}
                        />
                    </MotionWrapper>

                    <MotionWrapper delay={0.2} className="text-center">
                        <h3 className="text-sm uppercase tracking-widest opacity-60 mb-2">{t('dominant_emotion')}</h3>
                        <EmotionBadge
                            emotion={result.dominant_emotion}
                            size="lg"
                            className="shadow-xl"
                        />
                    </MotionWrapper>

                    {/* Audio Player Preview (Small) */}
                    {audioUrl && (
                        <div className="w-full mt-4 p-4 rounded-xl bg-gray-50 dark:bg-slate-900/50">
                            <div className="flex items-center gap-4">
                                <audio src={audioUrl} controls className="w-full h-8 opacity-80" />
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column: Detailed Breakdown */}
                <div className="space-y-6">
                    <MotionWrapper delay={0.3}>
                        <h3 className="text-lg font-semibold mb-4 border-l-4 border-indigo-500 pl-3">
                            {t('emotion_distribution')}
                        </h3>
                        <ProbabilityChart probabilities={result.emotions} />
                    </MotionWrapper>

                    {/* Placeholder for Word Timeline (Task 11) */}
                    {/* Word Timeline (Task 11) */}
                    {result.word_timestamps && result.word_timestamps.length > 0 && (
                        <MotionWrapper delay={0.4}>
                            <WordTimeline
                                wordTimestamps={result.word_timestamps}
                                audioDuration={Math.max(...result.word_timestamps.map(w => w.end))}
                            />
                        </MotionWrapper>
                    )}
                </div>
            </div>
        </div>
    );
};

const MotionWrapper: React.FC<{ children: React.ReactNode; delay?: number; className?: string }> = ({ children, delay = 0, className }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay }}
        className={className}
    >
        {children}
    </motion.div>
);

export default Result;
