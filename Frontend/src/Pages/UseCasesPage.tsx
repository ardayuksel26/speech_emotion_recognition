import React from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';
import { FaHeadset, FaChartLine, FaUserMd } from 'react-icons/fa';

const UseCasesPage = () => {
    const { t } = useTranslation();
    const { isDark } = useTheme();

    const cases = [
        {
            icon: <FaHeadset className="text-4xl text-indigo-500" />,
            title: t('use_case_call_center_title') || "Call Center Analytics",
            desc: t('use_case_call_center_desc') || "Monitor customer satisfaction in real-time and improve agent performance."
        },
        {
            icon: <FaUserMd className="text-4xl text-green-500" />,
            title: t('use_case_healthcare_title') || "Mental Health Monitoring",
            desc: t('use_case_healthcare_desc') || "Assist therapists in tracking emotional trends in patients over time."
        },
        {
            icon: <FaChartLine className="text-4xl text-blue-500" />,
            title: t('use_case_market_research_title') || "Market Research",
            desc: t('use_case_market_research_desc') || "Analyze emotional responses to products and advertisements."
        }
    ];

    return (
        <div className={`min-h-screen pt-20 px-6 ${isDark ? 'bg-slate-900 text-white' : 'bg-gray-50 text-gray-800'}`}>
            <div className="max-w-6xl mx-auto py-10">
                <h1 className="text-4xl font-bold mb-10 text-center">{t('use_cases')}</h1>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {cases.map((item, idx) => (
                        <div key={idx} className="bg-white/5 dark:bg-slate-800/50 p-8 rounded-2xl border border-gray-200 dark:border-slate-700 shadow-xl hover:scale-105 transition-transform duration-300">
                            <div className="mb-4">{item.icon}</div>
                            <h3 className="text-xl font-bold mb-2">{item.title}</h3>
                            <p className="opacity-80">{item.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default UseCasesPage;
