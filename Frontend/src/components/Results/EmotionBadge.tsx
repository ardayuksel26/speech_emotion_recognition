import React from 'react';
import { useTranslation } from 'react-i18next';
import { EMOTION_COLORS, EmotionType } from '../../constants/design';
import { useTheme } from '../../context/ThemeContext';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';

interface EmotionBadgeProps {
    emotion: string;
    confidence?: number;
    size?: 'sm' | 'md' | 'lg';
    className?: string;
}

const EmotionBadge: React.FC<EmotionBadgeProps> = ({
    emotion,
    confidence,
    size = 'md',
    className
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    const normalizedEmotion = (emotion?.toLowerCase() || 'neutral') as EmotionType;
    const colors = EMOTION_COLORS[normalizedEmotion] || EMOTION_COLORS.neutral;

    const currentStyle = isDark ? colors.dark : {
        bg: colors.bg,
        text: colors.text,
        border: colors.border
    };

    const sizeClasses = {
        sm: "px-2 py-0.5 text-xs",
        md: "px-4 py-1.5 text-sm",
        lg: "px-6 py-2 text-lg font-bold"
    };

    return (
        <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={clsx(
                "rounded-full border flex items-center gap-2 justify-center shadow-sm uppercase tracking-wider",
                sizeClasses[size],
                className
            )}
            style={{
                backgroundColor: currentStyle.bg,
                color: currentStyle.text,
                borderColor: currentStyle.border
            }}
        >
            <span>{t(`emotions.${normalizedEmotion}`)}</span>
            {confidence && (
                <span className="opacity-75 text-[0.8em]">
                    {Math.round(confidence * 100)}%
                </span>
            )}
        </motion.div>
    );
};

export default EmotionBadge;
