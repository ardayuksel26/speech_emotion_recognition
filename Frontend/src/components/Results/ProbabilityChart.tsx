import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { EmotionType } from '../../constants/design';
import { useTheme } from '../../context/ThemeContext';
import { clsx } from 'clsx';

interface ProbabilityChartProps {
    probabilities: Record<string, number>;
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probabilities }) => {
    const { t, i18n } = useTranslation();
    const { isDark } = useTheme();

    // Sort by probability desc, filter out zero-score emotions
    const sortedEmotions = Object.entries(probabilities)
        .filter(([, score]) => score > 0)
        .sort(([, a], [, b]) => b - a);

    return (
        <div className="w-full flex flex-col gap-2 md:gap-4 px-4 sm:px-10 py-8">
            {sortedEmotions.map(([emotion, score], index) => {
                const normalized = emotion.toLowerCase() as EmotionType | 'fear' | 'disgust' | 'surprise';

                const emotionStyles: Record<string, { bg: string, shadow: string }> = {
                    angry: { bg: 'bg-gradient-to-r from-rose-500 to-red-600', shadow: 'shadow-[0_0_12px_rgba(244,63,94,0.6)]' },
                    happy: { bg: 'bg-gradient-to-r from-amber-400 to-orange-500', shadow: 'shadow-[0_0_12px_rgba(251,191,36,0.6)]' },
                    sad: { bg: 'bg-gradient-to-r from-blue-400 to-indigo-500', shadow: 'shadow-[0_0_12px_rgba(96,165,250,0.6)]' },
                    neutral: { bg: 'bg-gradient-to-r from-slate-400 to-gray-500', shadow: 'shadow-[0_0_12px_rgba(148,163,184,0.6)]' },
                    calm: { bg: 'bg-gradient-to-r from-teal-400 to-emerald-500', shadow: 'shadow-[0_0_12px_rgba(45,212,191,0.6)]' },
                    fear: { bg: 'bg-gradient-to-r from-purple-500 to-violet-600', shadow: 'shadow-[0_0_12px_rgba(168,85,247,0.6)]' },
                    disgust: { bg: 'bg-gradient-to-r from-green-500 to-lime-600', shadow: 'shadow-[0_0_12px_rgba(34,197,94,0.6)]' },
                    surprise: { bg: 'bg-gradient-to-r from-pink-500 to-fuchsia-600', shadow: 'shadow-[0_0_12px_rgba(236,72,153,0.6)]' }
                };

                const style = emotionStyles[normalized] || { bg: 'bg-gradient-to-r from-indigo-500 to-purple-600', shadow: 'shadow-[0_0_12px_rgba(99,102,241,0.6)]' };

                return (
                    <div key={emotion} className="w-full px-2 sm:px-4">
                        <div className="flex justify-between items-end mb-2 px-2">
                            <span className={clsx(
                                "font-black tracking-widest text-sm md:text-base",
                                isDark ? "text-gray-200" : "text-gray-700"
                            )}>
                                {t(`emotions.${normalized}`).toLocaleUpperCase(i18n.language === 'tr' ? 'tr-TR' : 'en-US')}
                            </span>
                            <span className={clsx(
                                "font-bold text-sm md:text-base -mb-0.5",
                                isDark ? "text-gray-400" : "text-gray-500"
                            )}>
                                {(score * 100).toFixed(1)}%
                            </span>
                        </div>

                        <div className={clsx(
                            "w-full h-5 md:h-6 rounded-full overflow-hidden shadow-inner",
                            isDark ? "bg-slate-800/80" : "bg-gray-200/80"
                        )}>
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${score * 100}%` }}
                                transition={{ duration: 0.8, delay: index * 0.1, ease: 'easeOut' }}
                                className={clsx("h-full rounded-full", style.bg, style.shadow)}
                            />
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default ProbabilityChart;
