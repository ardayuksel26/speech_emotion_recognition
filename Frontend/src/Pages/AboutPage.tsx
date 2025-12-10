import React from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';

const AboutPage = () => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    return (
        <div className={`min-h-screen pt-20 px-6 ${isDark ? 'bg-slate-900 text-white' : 'bg-gray-50 text-gray-800'}`}>
            <div className="max-w-4xl mx-auto py-10">
                <h1 className="text-4xl font-bold mb-6">{t('about_us')}</h1>
                <div className="bg-white/5 dark:bg-slate-800/50 p-8 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-xl backdrop-blur-sm">
                    <p className="text-lg leading-relaxed mb-4">
                        {t('about_us_description_1') || "We are a team dedicated to exploring the intersection of AI and human emotion."}
                    </p>
                    <p className="text-lg leading-relaxed">
                        {t('about_us_description_2') || "Our project aims to bridge the gap between human expression and machine understanding through advanced Speech Emotion Recognition (SER) technologies."}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AboutPage;
