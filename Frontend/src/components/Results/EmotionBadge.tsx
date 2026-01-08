import React from 'react';
import { useTranslation } from 'react-i18next';
import { EMOTION_COLORS, EmotionType } from '../../constants/design';
import { useTheme } from '../../context/ThemeContext';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';

export const EMOTION_EMOJIS: Record<string, string> = {
    angry: '😠',
    calm: '😌',
    happy: '😊',
    sad: '😢',
    neutral: '😐',
    fear: 'a😨',
    disgust: '🤢',
    surprise: '😲'
};

interface EmotionBadgeProps {
    emotion: string;
    confidence?: number;
    size?: 'sm' | 'md' | 'lg' | 'xl';
    className?: string;
    showLabel?: boolean;
    showIcon?: boolean;
}

const EmotionBadge: React.FC<EmotionBadgeProps> = ({
    emotion,
    confidence,
    size = 'md',
    className,
    showLabel = true,
    showIcon = false
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    const normalizedEmotion = (emotion?.toLowerCase() || 'neutral') as EmotionType;
    const colors = EMOTION_COLORS[normalizedEmotion] || EMOTION_COLORS.neutral;
    const emoji = EMOTION_EMOJIS[normalizedEmotion] || '😐';

    // If showLabel is false, we default to showing the icon/emoji unless explicitly disabled
    const shouldShowIcon = showIcon || !showLabel;

    const currentStyle = isDark ? colors.dark : {
        bg: colors.bg,
        text: colors.text,
        border: colors.border
    };

    const sizeClasses = {
        sm: "px-2 py-0.5 text-xs",
        md: "px-4 py-1.5 text-sm",
        lg: "px-6 py-2 text-lg font-bold",
        xl: "px-0 py-0 text-5xl font-black bg-transparent border-0 shadow-none" // Special clean style for XL icon-only usage
    };

    return (
        <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className={clsx(
                "rounded-full flex items-center gap-2 justify-center shadow-sm uppercase tracking-wider transition-all",
                size !== 'xl' && "border", // Only add border for standard badges
                sizeClasses[size],
                className
            )}
            style={{
                backgroundColor: size === 'xl' ? 'transparent' : currentStyle.bg,
                color: currentStyle.text,
                borderColor: size === 'xl' ? 'transparent' : currentStyle.border
            }}
        >
            {shouldShowIcon && (
                <span className={clsx(
                    "leading-none",
                    { "mr-1": showLabel }
                )}>
                    {emoji}
                </span>
            )}

            {showLabel && (
                <span>{t(`emotions.${normalizedEmotion}`)}</span>
            )}

            {confidence && showLabel && (
                <span className="opacity-75 text-[0.8em]">
                    {Math.round(confidence * 100)}%
                </span>
            )}
        </motion.div>
    );
};

export default EmotionBadge;
