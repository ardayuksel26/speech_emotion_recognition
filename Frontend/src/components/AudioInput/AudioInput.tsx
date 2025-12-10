import React, { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaMicrophone, FaStop, FaTimes, FaFileAudio } from 'react-icons/fa';
import DragDropZone from './DragDropZone';
import WaveformVisualizer from './WaveformVisualizer';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface AudioInputProps {
    onAudioReady: (file: File) => void;
    className?: string;
}

const AudioInput: React.FC<AudioInputProps> = ({ onAudioReady, className }) => {
    const { t } = useTranslation();
    const [mode, setMode] = useState<'upload' | 'record' | 'preview'>('upload');
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
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
                setRecordedBlob(blob);
                const file = new File([blob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });
                onAudioReady(file);
                setMode('preview');

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            setMode('record');
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

    const cancelRecording = () => {
        stopRecording();
        setMode('upload');
        setRecordedBlob(null);
    };

    const handleFileSelect = (file: File) => {
        onAudioReady(file);
        // Logic to switch to preview or parent handles it?
        // Parent handles "Analysis" phase usually.
        // For now, we stay in 'upload' but maybe show selected state?
    };

    return (
        <div className={twMerge("w-full max-w-2xl mx-auto space-y-6", className)}>
            <div className="flex justify-center p-1 bg-gray-100 dark:bg-slate-800 rounded-full w-fit mx-auto mb-8 shadow-inner">
                <button
                    onClick={() => setMode('upload')}
                    className={clsx(
                        "flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-300",
                        mode === 'upload'
                            ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-md transform scale-105"
                            : "text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200"
                    )}
                >
                    <FaFileAudio className="text-lg" />
                    {t('upload_file')}
                </button>
                <button
                    onClick={() => !isRecording && setMode('record')}
                    className={clsx(
                        "flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-300",
                        mode === 'record'
                            ? "bg-white dark:bg-slate-700 text-indigo-600 dark:text-indigo-400 shadow-md transform scale-105"
                            : "text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200"
                    )}
                >
                    <FaMicrophone className="text-lg" />
                    {t('record_voice')}
                </button>
            </div>

            {mode === 'upload' && (
                <DragDropZone onFileSelect={handleFileSelect} />
            )}

            {mode === 'record' && (
                <div className="bg-white dark:bg-slate-800 rounded-3xl p-8 border border-slate-200 dark:border-slate-700 shadow-xl text-center">
                    <WaveformVisualizer
                        isRecording={isRecording}
                        recordingStream={streamRef.current || undefined}
                    />

                    <div className="mt-8 flex justify-center items-center gap-6">
                        {isRecording ? (
                            <button
                                onClick={stopRecording}
                                className="w-16 h-16 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center text-xl shadow-red-500/30 shadow-lg transition-transform hover:scale-110 active:scale-95"
                            >
                                <FaStop />
                            </button>
                        ) : (
                            <button
                                onClick={startRecording}
                                className="w-16 h-16 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white flex items-center justify-center text-xl shadow-indigo-500/30 shadow-lg transition-transform hover:scale-110 active:scale-95"
                            >
                                <FaMicrophone />
                            </button>
                        )}

                        {isRecording && (
                            <button
                                onClick={cancelRecording}
                                className="absolute top-4 right-4 text-slate-400 hover:text-slate-600"
                            >
                                <FaTimes />
                            </button>
                        )}
                    </div>

                    <p className="mt-4 text-sm text-slate-500">
                        {isRecording ? t('recording_in_progress') : t('click_mic_to_record')}
                    </p>
                </div>
            )}

            {mode === 'preview' && (
                <div className="text-center p-8 bg-green-50 dark:bg-green-900/10 rounded-2xl border border-green-100 dark:border-green-900/30">
                    <FaFileAudio className="mx-auto text-4xl text-green-500 mb-2" />
                    <p className="text-green-700 dark:text-green-300 font-medium">{t('audio_captured')}</p>
                    <button
                        onClick={() => setMode('upload')}
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
