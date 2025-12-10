import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { EMOTION_COLORS, EmotionType } from '../../constants/design';
import { useTheme } from '../../context/ThemeContext';
import { clsx } from 'clsx';

interface ProbabilityChartProps {
    probabilities: Record<string, number>;
}

const ProbabilityChart: React.FC<ProbabilityChartProps> = ({ probabilities }) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    // Sort by probability desc
    const sortedEmotions = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a);

    return (
        <div className="w-full space-y-3">
            {sortedEmotions.map(([emotion, score], index) => {
                const normalized = emotion.toLowerCase() as EmotionType;
                const color = EMOTION_COLORS[normalized] || EMOTION_COLORS.neutral;

                return (
                    <div key={emotion} className="w-full">
                        <div className="flex justify-between text-sm mb-1">
                            <span className={clsx(
                                "font-medium uppercase",
                                isDark ? "text-gray-300" : "text-gray-700"
                            )}>
                                {t(`emotions.${normalized}`)}
                            </span>
                            <span className={isDark ? "text-gray-400" : "text-gray-500"}>
                                {(score * 100).toFixed(1)}%
                            </span>
                        </div>

                        <div className={clsx(
                            "w-full h-3 rounded-full overflow-hidden",
                            isDark ? "bg-slate-700" : "bg-gray-200"
                        )}>
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${score * 100}%` }}
                                transition={{ duration: 0.8, delay: index * 0.1 }}
                                className="h-full rounded-full"
                                style={{ backgroundColor: color.primary }}
                            />
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default ProbabilityChart;
