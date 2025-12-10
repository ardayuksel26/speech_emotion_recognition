import React, { useEffect, useRef } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface WaveformVisualizerProps {
    audioUrl?: string;
    isRecording?: boolean;
    recordingStream?: MediaStream;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
    audioUrl,
    isRecording,
    recordingStream
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { isDark } = useTheme();
    const animationRef = useRef<number>();
    const audioContextRef = useRef<AudioContext>();
    const analyserRef = useRef<AnalyserNode>();
    const sourceRef = useRef<MediaStreamAudioSourceNode | MediaElementAudioSourceNode>();

    useEffect(() => {
        if (!canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Initialize Audio Context
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }

        // Canvas sizing
        const resizeCanvas = () => {
            canvas.width = canvas.parentElement?.clientWidth || 600;
            canvas.height = 120;
        };
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        // Drawing function
        const draw = () => {
            if (!ctx || !analyserRef.current) return;

            const bufferLength = analyserRef.current.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyserRef.current.getByteTimeDomainData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.lineWidth = 2;

            // Gradient
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, '#6366f1'); // Indigo-500
            gradient.addColorStop(0.5, '#a855f7'); // Purple-500
            gradient.addColorStop(1, '#ec4899'); // Pink-500
            ctx.strokeStyle = gradient;

            ctx.beginPath();
            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);

                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();

            animationRef.current = requestAnimationFrame(draw);
        };

        // Setup source based on props
        if (isRecording && recordingStream) {
            analyserRef.current = audioContextRef.current.createAnalyser();
            sourceRef.current = audioContextRef.current.createMediaStreamSource(recordingStream);
            sourceRef.current.connect(analyserRef.current);
            draw();
        } else if (audioUrl) {
            // For file playback visualization, we would attach to an audio element.
            // For now, we render a static "loaded" wave or setup an audio element context.
            // This implementation focuses on the recording visualization.
            // TODO: Implement file playback visualization (requires Audio Element ref).
        }

        return () => {
            window.removeEventListener('resize', resizeCanvas);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            sourceRef.current?.disconnect();
        };
    }, [isRecording, recordingStream, audioUrl, isDark]);

    return (
        <div className="w-full h-32 rounded-xl overflow-hidden bg-black/5 dark:bg-white/5 backdrop-blur-sm relative">
            <canvas ref={canvasRef} className="w-full h-full" />
        </div>
    );
};

export default WaveformVisualizer;
