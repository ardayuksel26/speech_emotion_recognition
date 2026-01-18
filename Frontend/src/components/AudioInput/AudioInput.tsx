import React, { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaMicrophone, FaStop, FaFileAudio, FaChevronLeft } from 'react-icons/fa';
import DragDropZone from './DragDropZone';
import WaveformVisualizer from './WaveformVisualizer';
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
}

const AudioInput: React.FC<AudioInputProps> = ({ onAudioReady, className }) => {
    const { t } = useTranslation();
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
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full animate-fadeIn">
                    <button
                        onClick={() => setMode('upload')}
                        className="group relative flex flex-col items-center justify-center p-10 h-80 rounded-[2rem] bg-white dark:bg-slate-800 border-2 border-transparent hover:border-indigo-500 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2"
                    >
                        <div className="w-24 h-24 rounded-full bg-indigo-50 dark:bg-indigo-900/30 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                            <FaFileAudio className="text-5xl text-indigo-600 dark:text-indigo-400" />
                        </div>
                        <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
                            {t('upload_file')}
                        </h3>
                        <p className="text-gray-500 dark:text-slate-400 text-center px-4">
                            WAV, MP3, OGG formatındaki ses dosyalarını yükle
                        </p>
                    </button>

                    <button
                        onClick={() => setMode('record')}
                        className="group relative flex flex-col items-center justify-center p-10 h-80 rounded-[2rem] bg-white dark:bg-slate-800 border-2 border-transparent hover:border-rose-500 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2"
                    >
                        <div className="w-24 h-24 rounded-full bg-rose-50 dark:bg-rose-900/30 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                            <FaMicrophone className="text-5xl text-rose-600 dark:text-rose-400" />
                        </div>
                        <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
                            {t('record_voice')}
                        </h3>
                        <p className="text-gray-500 dark:text-slate-400 text-center px-4">
                            Mikrofonu kullanarak anlık ses kaydı al
                        </p>
                    </button>
                </div>
            )}

            {/* Upload Mode */}
            {mode === 'upload' && (
                <div className="animate-fadeIn relative">
                    <div className="bg-white dark:bg-slate-800 rounded-[2.5rem] p-8 md:p-10 border border-slate-200 dark:border-slate-700 shadow-2xl relative overflow-hidden">
                        <button
                            onClick={goBack}
                            className="absolute top-4 left-4 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 bg-white/80 dark:bg-white/10 backdrop-blur-md shadow-sm hover:shadow-md hover:scale-105 text-slate-600 dark:text-slate-200 z-10"
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
                    <div className="bg-white dark:bg-slate-800 rounded-[2.5rem] p-10 border border-slate-200 dark:border-slate-700 shadow-2xl text-center relative overflow-hidden">

                        {!isRecording && (
                            <button
                                onClick={goBack}
                                className="absolute top-4 left-4 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 bg-white/80 dark:bg-white/10 backdrop-blur-md shadow-sm hover:shadow-md hover:scale-105 text-slate-600 dark:text-slate-200 z-10"
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
