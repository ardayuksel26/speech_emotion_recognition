import React from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';

const TechnicalInfoPage = () => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    return (
        <div className={`min-h-screen pt-20 px-6 ${isDark ? 'bg-slate-900 text-white' : 'bg-gray-50 text-gray-800'}`}>
            <div className="max-w-5xl mx-auto py-10">
                <h1 className="text-4xl font-bold mb-8">{t('technical_info') || "Technical Information"}</h1>

                <div className="space-y-8">
                    {/* Model Architecture */}
                    <div className="bg-white/5 dark:bg-slate-800/50 p-8 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-lg">
                        <h2 className="text-2xl font-semibold mb-4 text-indigo-500">{t('tech_model_title') || "Model Architecture"}</h2>
                        <ul className="list-disc list-inside space-y-2 opacity-80">
                            <li><strong>Feature Extraction:</strong> MFCCs, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz.</li>
                            <li><strong>Model:</strong> Random Forest Classifier (optimized for tabular feature data).</li>
                            <li><strong>Training:</strong> Segment-level training with majority voting aggregation.</li>
                        </ul>
                    </div>

                    {/* Backend Stack */}
                    <div className="bg-white/5 dark:bg-slate-800/50 p-8 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-lg">
                        <h2 className="text-2xl font-semibold mb-4 text-green-500">{t('tech_backend_title') || "Backend Technology"}</h2>
                        <ul className="list-disc list-inside space-y-2 opacity-80">
                            <li><strong>Framework:</strong> FastAPI (Python)</li>
                            <li><strong>Audio Processing:</strong> Librosa, Pydub</li>
                            <li><strong>Asynchronous Processing:</strong> BackgroundTasks for non-blocking analysis</li>
                        </ul>
                    </div>

                    {/* Frontend Stack */}
                    <div className="bg-white/5 dark:bg-slate-800/50 p-8 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-lg">
                        <h2 className="text-2xl font-semibold mb-4 text-pink-500">{t('tech_frontend_title') || "Frontend Technology"}</h2>
                        <ul className="list-disc list-inside space-y-2 opacity-80">
                            <li><strong>Framework:</strong> React + Vite (TypeScript)</li>
                            <li><strong>State Management:</strong> Context API</li>
                            <li><strong>Styling:</strong> TailwindCSS, Framer Motion</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TechnicalInfoPage;
