export interface AudioFile {
    file: File;
    previewUrl: string;
    duration?: number;
}

export interface AnalysisResult {
    emotions: {
        happy: number;
        sad: number;
        angry: number;
        calm: number;
        [key: string]: number;
    };
    dominant_emotion: string;
    confidence: number;
    transcript?: string;
    word_timestamps?: Array<{
        word: string;
        start: number;
        end: number;
        emotion: string;
        confidence: number;
    }>;
}

export interface ProcessingStatus {
    step: 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error';
    progress: number;
    message?: string;
}
