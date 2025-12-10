
import { AnalysisResult } from '../types';

export const generateFilename = (type: 'json' | 'csv', prefix: string = 'analysis_result') => {
    const date = new Date();
    const timestamp = date.toISOString().replace(/[:.]/g, '-');
    return `${prefix}_${timestamp}.${type}`;
};

export const downloadFile = (content: string, filename: string, contentType: string) => {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

export const exportToJSON = (data: AnalysisResult, filename?: string) => {
    const jsonContent = JSON.stringify(data, null, 2);
    const name = filename || generateFilename('json');
    downloadFile(jsonContent, name, 'application/json');
};

export const exportToCSV = (data: AnalysisResult, filename?: string) => {
    // Flatten the data for CSV
    // Header
    const headers = ['Word', 'Start Time (s)', 'End Time (s)', 'Emotion', 'Confidence', 'Sentence Dominant Emotion'];

    // Rows
    const rows = data.word_timestamps?.map(word => [
        word.word || 'unknown',
        word.start.toFixed(3),
        word.end.toFixed(3),
        word.emotion,
        word.confidence.toFixed(3),
        data.dominant_emotion
    ]) || [];

    // Combine
    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');

    const name = filename || generateFilename('csv');
    downloadFile(csvContent, name, 'text/csv;charset=utf-8;');
};
