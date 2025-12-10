import React from 'react';
import { useTheme } from '../../context/ThemeContext';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { EMOTION_COLORS, EmotionType } from '../../constants/design';

interface ConfidenceMeterProps {
    value: number; // 0 to 1
    emotion: string;
    size?: number;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({ value, emotion, size = 120 }) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const radius = size * 0.4;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (value * circumference);

    const normalizedEmotion = (emotion?.toLowerCase() || 'neutral') as EmotionType;
    const color = EMOTION_COLORS[normalizedEmotion]?.primary || EMOTION_COLORS.neutral.primary;

    return (
        <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="transform -rotate-90">
                {/* Background Circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}
                    strokeWidth="10"
                    fill="transparent"
                />
                {/* Progress Circle */}
                <motion.circle
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={color}
                    strokeWidth="10"
                    fill="transparent"
                    strokeDasharray={circumference}
                    strokeLinecap="round"
                />
            </svg>
            <div className="absolute flex flex-col items-center">
                <span className={`text-3xl font-bold ${isDark ? "text-white" : "text-gray-800"}`}>
                    {Math.round(value * 100)}%
                </span>
                <span className={`text-xs uppercase tracking-widest ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                    {t('confidence')}
                </span>
            </div>
        </div>
    );
};

export default ConfidenceMeter;
