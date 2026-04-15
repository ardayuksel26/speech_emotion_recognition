import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { AnalysisResult } from '../../types';
import { EMOTION_COLORS } from '../../constants/design';
import clsx from 'clsx';
import { useTheme } from '../../context/ThemeContext';
import { useTranslation } from 'react-i18next';

interface WordTimelineProps {
    wordTimestamps: NonNullable<AnalysisResult['word_timestamps']>;
    onWordClick?: (index: number) => void;
    audioDuration: number;
    dominantEmotion: string;
}

export const WordTimeline: React.FC<WordTimelineProps> = ({
    wordTimestamps,
    onWordClick,
    audioDuration,
    dominantEmotion,
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    if (!wordTimestamps || wordTimestamps.length === 0) return null;

    const totalDuration = audioDuration || Math.max(...wordTimestamps.map(w => w.end));
    const color = EMOTION_COLORS[dominantEmotion as keyof typeof EMOTION_COLORS]?.primary || '#6366f1';

    return (
        <div className="w-full space-y-4">
            <div
                ref={scrollContainerRef}
                className="relative w-full overflow-x-auto custom-scrollbar"
            >
                {/* Timeline Container — bar alanı + confidence alanı + zaman işaretleri */}
                <div
                    className={clsx(
                        "relative rounded-xl border w-full",
                        isDark ? "bg-slate-800/80 border-slate-700/50 shadow-inner" : "bg-gray-50 border-gray-100"
                    )}
                    style={{ height: '100px' }}
                >
                    {/* Zaman işaretleri (en alt) */}
                    <div className={clsx(
                        "absolute bottom-0 left-0 w-full h-5 border-t flex justify-between px-2 text-[10px] font-mono items-center",
                        isDark ? "border-slate-700 text-slate-400" : "border-gray-200 text-gray-400"
                    )}>
                        <span>0.0s</span>
                        <span>{(totalDuration / 2).toFixed(1)}s</span>
                        <span>{totalDuration.toFixed(1)}s</span>
                    </div>

                    {/* Segmentler — bar (h-10) + confidence text (h-5) */}
                    <div className="absolute top-3 left-0 w-full px-1" style={{ height: '72px' }}>
                        {wordTimestamps.map((word, index) => {
                            const left = (word.start / totalDuration) * 100;
                            const width = Math.max((word.end - word.start) / totalDuration * 100, 1.5);
                            const conf = typeof word.confidence === 'number' ? word.confidence : 1;

                            return (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, scaleY: 0 }}
                                    animate={{ opacity: 1, scaleY: 1 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="absolute group cursor-pointer flex flex-col items-center"
                                    style={{ left: `${left}%`, width: `${width}%`, top: 0, height: '72px' }}
                                    onClick={() => onWordClick?.(index)}
                                >
                                    {/* Bar */}
                                    <div
                                        className="w-full rounded-md border transition-all duration-200 hover:brightness-125 hover:shadow-lg relative"
                                        style={{
                                            height: '40px',
                                            backgroundColor: `${color}${Math.round(conf * 80 + 20).toString(16).padStart(2, '0')}`,
                                            borderColor: color,
                                            borderWidth: '1px',
                                        }}
                                    >
                                        <div
                                            className="w-full h-0.5 absolute top-1/2 -translate-y-1/2 rounded-full opacity-70"
                                            style={{ backgroundColor: color }}
                                        />
                                    </div>

                                    {/* Confidence altında */}
                                    <div
                                        className="mt-1 text-[9px] font-bold leading-none text-center w-full truncate"
                                        style={{ color }}
                                    >
                                        {(conf * 100).toFixed(0)}%
                                    </div>

                                    {/* Kelime */}
                                    {word.word && (
                                        <div
                                            className="text-[8px] leading-none text-center w-full truncate"
                                            style={{ color: isDark ? '#94a3b8' : '#64748b' }}
                                        >
                                            {word.word}
                                        </div>
                                    )}

                                    {/* Tooltip */}
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-20" style={{ minWidth: '110px' }}>
                                        <div className="rounded-lg py-2 px-3 shadow-xl text-white text-xs flex flex-col gap-1" style={{ background: 'rgba(15,23,42,0.97)', border: `1px solid ${color}60` }}>
                                            <span className="font-black capitalize text-center text-sm" style={{ color }}>
                                                {t((dominantEmotion || 'neutral').toLowerCase())}
                                            </span>
                                            <span className="text-center font-mono text-[11px] text-slate-300">
                                                {word.start.toFixed(2)}s – {word.end.toFixed(2)}s
                                            </span>
                                            <span className="text-center font-bold text-[13px] text-white">
                                                %{(conf * 100).toFixed(1)}
                                            </span>
                                            {word.word && (
                                                <span className="text-center font-bold text-[13px] text-white border-t pt-1" style={{ borderColor: `${color}40` }}>
                                                    "{word.word}"
                                                </span>
                                            )}
                                            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent" style={{ borderTopColor: 'rgba(15,23,42,0.97)' }} />
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Renk Açıklaması */}
            <div className="flex flex-wrap gap-3 pt-1">
                {(['happy', 'calm', 'angry', 'sad'] as const).map((emotion) => (
                    <div key={emotion} className="flex items-center gap-1.5">
                        <span className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: EMOTION_COLORS[emotion].primary }} />
                        <span className={clsx("text-xs font-semibold capitalize", isDark ? "text-slate-400" : "text-slate-500")}>
                            {t(emotion)}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
};
