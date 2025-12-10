import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResult } from '../../types';
import { EMOTION_COLORS } from '../../constants/design';
import clsx from 'clsx';

interface WordTimelineProps {
    wordTimestamps: NonNullable<AnalysisResult['word_timestamps']>;
    onWordClick?: (index: number) => void;
    audioDuration: number;
}

import { useTranslation } from 'react-i18next';

export const WordTimeline: React.FC<WordTimelineProps> = ({
    wordTimestamps,
    onWordClick,
    audioDuration
}) => {
    const { t } = useTranslation();
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    if (!wordTimestamps || wordTimestamps.length === 0) return null;

    // Calculate total width based on duration or min-width
    // We want the timeline to be proportional if it fits, or scrollable if long
    const totalDuration = audioDuration || Math.max(...wordTimestamps.map(w => w.end));

    return (
        <div className="w-full space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
                    {t('word_by_word_analysis')}
                </h3>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                    {t('click_segment_details')}
                </span>
            </div>

            <div
                ref={scrollContainerRef}
                className="relative w-full overflow-x-auto pb-6 custom-scrollbar"
            >
                {/* Timeline Container */}
                <div
                    className="relative h-24 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-100 dark:border-gray-700/50 min-w-[600px]"
                    style={{ width: '100%' }}
                >
                    {/* Time Markers */}
                    <div className="absolute bottom-0 left-0 w-full h-6 border-t border-gray-200 dark:border-gray-700 flex justify-between px-2 text-[10px] text-gray-400 font-mono">
                        <span>0.0s</span>
                        <span>{(totalDuration / 2).toFixed(1)}s</span>
                        <span>{totalDuration.toFixed(1)}s</span>
                    </div>

                    {/* Segments */}
                    <div className="absolute top-4 left-0 w-full h-12 flex items-center px-1">
                        {wordTimestamps.map((word, index) => {
                            // Calculate position and width as percentage of total duration
                            const left = (word.start / totalDuration) * 100;
                            const width = ((word.end - word.start) / totalDuration) * 100;
                            const color = EMOTION_COLORS[word.emotion as keyof typeof EMOTION_COLORS]?.primary || '#9CA3AF';

                            return (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, scaleY: 0 }}
                                    animate={{ opacity: 1, scaleY: 1 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="absolute h-full group cursor-pointer"
                                    style={{
                                        left: `${left}%`,
                                        width: `${width}%`,
                                        minWidth: '4px' // Ensure visibility for very short segments
                                    }}
                                    onClick={() => onWordClick?.(index)}
                                >
                                    {/* Segment Bar */}
                                    <div
                                        className={clsx(
                                            "w-full h-full rounded-md border transition-all duration-200",
                                            "hover:brightness-110 hover:shadow-lg hover:z-10 hover:scale-y-110"
                                        )}
                                        style={{
                                            backgroundColor: `${color}40`, // 25% opacity
                                            borderColor: color,
                                            borderWidth: '1px'
                                        }}
                                    >
                                        {/* Inner Solid Line for emphasis */}
                                        <div
                                            className="w-full h-1 absolute top-1/2 -translate-y-1/2 rounded-full opacity-60"
                                            style={{ backgroundColor: color }}
                                        />
                                    </div>

                                    {/* Tooltip */}
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max hidden group-hover:block z-20">
                                        <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 shadow-xl backdrop-blur-sm bg-opacity-90">
                                            <div className="font-bold mb-1 capitalize text-center" style={{ color }}>
                                                {word.emotion}
                                            </div>
                                            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] opacity-90">
                                                <span>Time:</span>
                                                <span className="text-right font-mono">{word.start.toFixed(2)}s - {word.end.toFixed(2)}s</span>
                                                <span>Conf:</span>
                                                <span className="text-right font-mono">{(word.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                            {/* Arrow */}
                                            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900 opacity-90"></div>
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};
