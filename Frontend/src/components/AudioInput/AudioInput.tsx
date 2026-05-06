import React, { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaMicrophone, FaStop, FaFileAudio, FaChevronLeft } from 'react-icons/fa';
import DragDropZone from './DragDropZone';
import WaveformVisualizer from './WaveformVisualizer';
import { useTheme } from '../../context/ThemeContext';
// clsx import removed (unused)
// clsx is unused, removing or keeping if intended for future? Error says unused.
// actually, let's just remove the import if it's the only usage.
// Wait, is it used? "clsx is declared but its value is never read."
// Checking file content from previous turn...
// Line 6: import { clsx } from 'clsx';
// Line 7: import { twMerge } from 'tailwind-merge';
// Line 76: <div className={twMerge("w-full max-w-4xl mx-auto", className)}>
// It seems clsx is NOT used.

import { twMerge } from 'tailwind-merge';

interface AudioInputProps {
    onAudioReady: (file: File) => void;
    className?: string;
    compact?: boolean;
}

const AudioInput: React.FC<AudioInputProps> = ({ onAudioReady, className, compact = false }) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
    const [mode, setMode] = useState<'initial' | 'upload' | 'record' | 'preview'>('initial');
    const [isRecording, setIsRecording] = useState(false);
    // const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null); // Unused
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const chunksRef = useRef<BlobPart[]>([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;
            mediaRecorderRef.current = new MediaRecorder(stream);
            chunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
                // setRecordedBlob(blob); // Unused state
                const file = new File([blob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });
                onAudioReady(file);
                // Parent will likely unmount or change view, but if not:
                setMode('preview');

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert(t('microphone_permission_denied') || 'Microphone access denied');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleFileSelect = (file: File) => {
        onAudioReady(file);
    };

    const goBack = () => {
        setMode('initial');
    };

    return (
        <div className={twMerge("w-full max-w-4xl mx-auto", className)}>

            {/* Initial Selection Mode */}
            {mode === 'initial' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 md:gap-8 w-full animate-fadeIn">
                    <button
                        onClick={() => setMode('upload')}
                        className={`audio-card ${compact ? 'audio-card-compact' : ''} group relative flex flex-col items-center justify-center p-4 sm:p-6 md:p-7 rounded-[2.5rem] border border-slate-200 dark:border-white/10 hover:border-indigo-400 dark:hover:border-indigo-500 shadow-[0_8px_32px_rgba(0,0,0,0.08)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.2)] hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 overflow-hidden`}
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/0 to-purple-500/0 group-hover:from-indigo-500/8 group-hover:to-purple-500/8 transition-colors duration-500 pointer-events-none" />
                        <div className="audio-icon-box w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 rounded-2xl flex items-center justify-center mb-3 sm:mb-5 md:mb-6 group-hover:scale-110 group-hover:-rotate-6 transition-transform duration-500 shadow-sm">
                            <FaFileAudio className="text-2xl sm:text-3xl md:text-4xl text-indigo-500 dark:text-indigo-400 drop-shadow-md" />
                        </div>
                        <h3 
                            className="text-lg sm:text-xl md:text-2xl font-bold mb-1 sm:mb-2 md:mb-3 tracking-tight transition-colors"
                            style={{ color: isDark ? '#ffffff' : '#000000' }}
                        >
                            {t('upload_audio')}
                        </h3>
                        <p 
                            className="text-center text-xs sm:text-sm px-2 sm:px-4 leading-relaxed font-medium hidden sm:block"
                            style={{ color: isDark ? '#cbd5e1' : '#334155' }}
                        >
                            {t('upload_desc')}
                        </p>
                    </button>

                    <button
                        onClick={() => setMode('record')}
                        className={`audio-card ${compact ? 'audio-card-compact' : ''} group relative flex flex-col items-center justify-center p-4 sm:p-6 md:p-7 rounded-[2.5rem] border border-slate-200 dark:border-white/10 hover:border-rose-400 dark:hover:border-rose-500 shadow-[0_8px_32px_rgba(0,0,0,0.08)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.2)] hover:shadow-2xl transition-all duration-500 hover:-translate-y-2 overflow-hidden`}
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-rose-500/0 to-orange-500/0 group-hover:from-rose-500/8 group-hover:to-orange-500/8 transition-colors duration-500 pointer-events-none" />
                        <div className="audio-icon-box w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 rounded-2xl flex items-center justify-center mb-3 sm:mb-5 md:mb-6 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-500 shadow-sm">
                            <FaMicrophone className="text-2xl sm:text-3xl md:text-4xl text-rose-500 dark:text-rose-400 drop-shadow-md" />
                        </div>
                        <h3 
                            className="text-lg sm:text-xl md:text-2xl font-bold mb-1 sm:mb-2 md:mb-3 tracking-tight transition-colors"
                            style={{ color: isDark ? '#ffffff' : '#000000' }}
                        >
                            {t('record_audio')}
                        </h3>
                        <p 
                            className="text-center text-xs sm:text-sm px-2 sm:px-4 leading-relaxed font-medium hidden sm:block"
                            style={{ color: isDark ? '#cbd5e1' : '#334155' }}
                        >
                            {t('record_desc')}
                        </p>
                    </button>
                </div>
            )}

            {/* Upload Mode */}
            {mode === 'upload' && (
                <div className="animate-fadeIn relative">
                    <div className="audio-panel rounded-[2.5rem] p-8 md:p-10 border border-slate-200 dark:border-slate-700 shadow-2xl relative overflow-hidden">
                        <button
                            onClick={goBack}
                            className={`absolute top-4 left-4 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 backdrop-blur-md shadow-sm hover:shadow-md hover:scale-105 z-10 ${isDark ? 'bg-white/10 text-slate-200 hover:bg-white/20' : 'bg-white text-slate-900 hover:bg-slate-100'}`}
                        >
                            <FaChevronLeft className="text-sm" />
                        </button>

                        <div className="mt-8">
                            <DragDropZone onFileSelect={handleFileSelect} />
                        </div>
                    </div>
                </div>
            )}

            {/* Record Mode */}
            {mode === 'record' && (
                <div className="animate-fadeIn relative">
                    <div className="audio-panel rounded-[2.5rem] p-10 border border-slate-200 dark:border-slate-700 shadow-2xl text-center relative overflow-hidden">

                        {!isRecording && (
                            <button
                                onClick={goBack}
                                className={`absolute top-4 left-4 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 backdrop-blur-md shadow-sm hover:shadow-md hover:scale-105 z-10 ${isDark ? 'bg-white/10 text-slate-200 hover:bg-white/20' : 'bg-white text-slate-900 hover:bg-slate-100'}`}
                            >
                                <FaChevronLeft className="text-sm" />
                            </button>
                        )}

                        <WaveformVisualizer
                            isRecording={isRecording}
                            recordingStream={streamRef.current || undefined}
                        />

                        <div className="mt-10 flex justify-center items-center gap-8 relative z-10">
                            {isRecording ? (
                                <button
                                    onClick={stopRecording}
                                    className="group relative w-20 h-20 rounded-full flex items-center justify-center bg-rose-500 hover:bg-rose-600 text-white shadow-lg shadow-rose-500/40 transition-all hover:scale-105 active:scale-95"
                                >
                                    <FaStop className="text-2xl" />
                                    <span className="absolute -bottom-10 text-slate-500 dark:text-slate-400 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                        Kaydı Bitir
                                    </span>
                                </button>
                            ) : (
                                <button
                                    onClick={startRecording}
                                    className="group relative w-20 h-20 rounded-full flex items-center justify-center bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/40 transition-all hover:scale-105 active:scale-95"
                                >
                                    <FaMicrophone className="text-3xl" />
                                    <span className="absolute -bottom-10 text-slate-500 dark:text-slate-400 text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                        Kaydı Başlat
                                    </span>
                                </button>
                            )}

                            {/* Cancel Button Removed */}
                        </div>

                        <p className="mt-8 text-slate-500 font-medium">
                            {isRecording ? (
                                <span className="flex items-center justify-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                                    {t('recording_in_progress')}
                                </span>
                            ) : t('click_mic_to_record')}
                        </p>
                    </div>
                </div>
            )}

            {mode === 'preview' && (
                <div className="text-center p-8 bg-green-50 dark:bg-green-900/10 rounded-2xl border border-green-100 dark:border-green-900/30 animate-fadeIn">
                    <FaFileAudio className="mx-auto text-4xl text-green-500 mb-2" />
                    <p className="text-green-700 dark:text-green-300 font-medium">{t('audio_captured')}</p>
                    <button
                        onClick={() => setMode('initial')}
                        className="mt-4 text-sm underline text-slate-500 hover:text-indigo-500"
                    >
                        {t('try_again')}
                    </button>
                </div>
            )}
        </div>
    );
};

export default AudioInput;
