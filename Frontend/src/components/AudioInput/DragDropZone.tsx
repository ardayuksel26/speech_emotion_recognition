import React, { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCloudUploadAlt, FaMusic } from 'react-icons/fa';
import { useTheme } from '../../context/ThemeContext';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface DragDropZoneProps {
    onFileSelect: (file: File) => void;
    className?: string;
    disabled?: boolean;
}

const DragDropZone: React.FC<DragDropZoneProps> = ({
    onFileSelect,
    className,
    disabled = false
}) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
    }, [disabled]);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (disabled) return;

        const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith('audio/')) {
            onFileSelect(file);
        } else {
            // Find a better way to show error than alert
            console.warn('Invalid file type');
        }
    }, [disabled, onFileSelect]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) onFileSelect(file);
    };

    return (
        <div
            onClick={() => !disabled && fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={twMerge(
                clsx(
                    'relative overflow-hidden rounded-[2.5rem] border-2 border-dashed transition-all duration-500 cursor-pointer group',
                    'flex flex-col items-center justify-center p-10 min-h-[300px]',
                    isDark
                        ? 'bg-slate-800/40 border-slate-600/50 hover:border-indigo-400 backdrop-blur-xl shadow-lg'
                        : 'bg-white/40 border-indigo-200 hover:border-indigo-500 backdrop-blur-xl shadow-lg',
                    isDragging && (isDark ? 'bg-indigo-500/20 border-indigo-400 shadow-[0_0_30px_rgba(99,102,241,0.2)]' : 'bg-indigo-50/80 border-indigo-500 shadow-[0_0_30px_rgba(99,102,241,0.15)]'),
                    disabled && 'opacity-50 cursor-not-allowed grayscale'
                ),
                className
            )}
        >
            <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

            <div className={clsx(
                "relative w-24 h-24 rounded-full flex items-center justify-center mb-6 transition-transform duration-500 group-hover:scale-110 shadow-xl",
                isDark ? "bg-slate-700/50" : "bg-indigo-50"
            )}>
                {isDragging ? (
                    <FaMusic className="text-4xl text-indigo-500 animate-bounce" />
                ) : (
                    <FaCloudUploadAlt className="text-5xl text-indigo-500" />
                )}
            </div>

            <h3 className={clsx(
                "text-2xl font-bold mb-3 text-center transition-colors",
                isDark ? "text-slate-200" : "text-slate-800"
            )}>
                {isDragging ? t('drop_audio_here') : t('drag_drop_audio')}
            </h3>

            <p className={clsx(
                "text-sm text-center max-w-xs leading-relaxed",
                isDark ? "text-slate-400" : "text-slate-500"
            )}>
                {t('supported_formats_desc') || 'WAV, MP3, OGG up to 10MB'}
            </p>

            <button className={clsx(
                "mt-8 px-6 py-2.5 rounded-xl font-medium text-sm transition-all shadow-lg hover:shadow-indigo-500/25",
                "bg-indigo-600 text-white hover:bg-indigo-700 active:scale-95"
            )}>
                {t('browse_files')}
            </button>

            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleChange}
                className="hidden"
                disabled={disabled}
            />
        </div>
    );
};

export default DragDropZone;
