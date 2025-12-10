import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaMoon, FaSun, FaChevronDown, FaBars, FaTimes, FaInfoCircle, FaBullseye, FaGlobe, FaCogs } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";
import { Link, useLocation } from "react-router-dom";
import { clsx } from "clsx";

const Header = () => {
  const { t, i18n } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const [isLangOpen, setIsLangOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const languages = {
    en: { label: "English", flagUrl: "https://flagcdn.com/gb.svg" },
    tr: { label: "Türkçe", flagUrl: "https://flagcdn.com/tr.svg" },
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsLangOpen(false);
  };

  const currentLanguage = i18n.language.split('-')[0];

  const navLinks = [
    { key: 'about_us', path: '/about', icon: <FaInfoCircle /> },
    { key: 'use_cases', path: '/use-cases', icon: <FaBullseye /> },
    { key: 'technical_info', path: '/technical-info', icon: <FaCogs /> },
  ];

  return (
    <header className={clsx(
      "relative w-full shadow-xl sticky top-0 z-30 transition-all duration-300 flex items-center h-16 font-sans px-4 md:px-8 justify-between",
      isDark ? "bg-slate-900 text-white border-b border-slate-700" : "bg-gradient-to-r from-indigo-100 to-purple-100 text-gray-800 border-b border-indigo-200"
    )}>

      {/* LEFT: Logo / Title */}
      <Link to="/" className="z-20 flex items-center gap-2 group ml-4 md:ml-12">
        <h1 className={clsx(
          "text-lg md:text-xl font-extrabold tracking-tight whitespace-nowrap drop-shadow-sm transition-transform group-hover:scale-105",
          isDark ? "text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-indigo-200" : "text-gray-800"
        )}>
          {t('title')}
        </h1>
      </Link>

      {/* CENTER: Desktop Navigation */}
      <nav className="hidden md:flex items-center gap-6 absolute left-1/2 transform -translate-x-1/2">
        {navLinks.map((link) => (
          <Link
            key={link.key}
            to={link.path}
            className={clsx(
              "text-sm font-semibold transition-colors duration-200 flex items-center gap-2",
              location.pathname === link.path
                ? (isDark ? "text-indigo-300" : "text-indigo-600")
                : (isDark ? "text-slate-300 hover:text-white" : "text-gray-600 hover:text-indigo-600")
            )}
          >
            {t(link.key)}
          </Link>
        ))}
      </nav>

      {/* RIGHT: Controls (Lang, Theme, Hamburger) */}
      <div className="flex items-center gap-3 z-20">

        {/* Language Switcher (Visible on Mobile & Desktop per request) */}
        <div className="relative">
          <button
            onClick={() => setIsLangOpen(!isLangOpen)}
            className={clsx(
              "flex items-center gap-2 rounded-full px-3 py-1.5 shadow-sm hover:shadow-md transition-all duration-200",
              isDark ? "bg-slate-800 text-white hover:bg-slate-700" : "bg-white text-indigo-700 hover:bg-indigo-50"
            )}
          >
            <img
              src={languages[currentLanguage]?.flagUrl}
              alt={languages[currentLanguage]?.label}
              className="w-5 h-5 rounded-full object-cover"
            />
            <span className="hidden md:inline text-sm font-bold">
              {currentLanguage.toUpperCase()}
            </span>
            <FaChevronDown className={clsx("text-xs transition-transform duration-200", isLangOpen && "rotate-180")} />
          </button>

          {isLangOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setIsLangOpen(false)}></div>
              <div className="absolute right-0 mt-2 w-40 bg-white text-slate-800 rounded-2xl shadow-xl py-2 z-50 animate-fade-in-down">
                {Object.keys(languages).map((code) => (
                  <button
                    key={code}
                    onClick={() => changeLanguage(code)}
                    className="w-full flex items-center justify-between px-4 py-3 hover:bg-blue-50 text-sm transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <img src={languages[code].flagUrl} alt="" className="w-5 h-5 rounded-full" />
                      <span className="font-medium">{languages[code].label}</span>
                    </div>
                    {currentLanguage === code && <span className="text-green-500 font-bold">✓</span>}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Theme Switcher (Visible on Mobile & Desktop per request) */}
        <button
          onClick={toggleTheme}
          className={clsx(
            "relative w-12 h-6 rounded-full transition-colors duration-300 focus:outline-none border border-white/30 shadow-inner",
            isDark ? "bg-slate-800" : "bg-indigo-100"
          )}
        >
          <div className={clsx(
            "absolute top-0.5 left-0.5 w-5 h-5 rounded-full shadow-md transform transition-transform duration-300 flex items-center justify-center text-[10px]",
            isDark ? "translate-x-6 bg-slate-700 text-indigo-300" : "translate-x-0 bg-white text-amber-500"
          )}>
            {isDark ? <FaMoon /> : <FaSun />}
          </div>
        </button>

        {/* Mobile Hamburger (Hidden on Desktop) */}
        <button
          onClick={() => setIsMenuOpen(true)}
          className={clsx(
            "md:hidden p-2 rounded-lg transition-all duration-200",
            isDark ? "text-white hover:bg-slate-800" : "text-indigo-700 hover:bg-indigo-50"
          )}
        >
          <FaBars className="text-xl" />
        </button>
      </div>

      {/* Mobile Menu Drawer */}
      <div className={clsx(
        "fixed inset-0 z-50 transform transition-transform duration-300 md:hidden",
        isMenuOpen ? "translate-x-0" : "translate-x-full"
      )}>
        {/* Backdrop */}
        <div
          className={clsx("absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity", isMenuOpen ? "opacity-100" : "opacity-0 pointer-events-none")}
          onClick={() => setIsMenuOpen(false)}
        />

        {/* Drawer Content */}
        <div className={clsx(
          "absolute right-0 h-full w-64 shadow-2xl flex flex-col p-6 transition-colors duration-300",
          isDark ? "bg-slate-900 border-l border-slate-800" : "bg-white border-l border-gray-100"
        )}>
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-lg font-bold">{t('title')}</h2>
            <button onClick={() => setIsMenuOpen(false)} className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-slate-800">
              <FaTimes />
            </button>
          </div>

          <nav className="flex flex-col gap-4">
            {navLinks.map((link) => (
              <Link
                key={link.key}
                to={link.path}
                onClick={() => setIsMenuOpen(false)}
                className={clsx(
                  "flex items-center gap-3 px-4 py-3 rounded-xl transition-all",
                  location.pathname === link.path
                    ? (isDark ? "bg-indigo-500/20 text-indigo-300" : "bg-indigo-50 text-indigo-600")
                    : (isDark ? "text-slate-300 hover:bg-white/5" : "text-gray-600 hover:bg-gray-50")
                )}
              >
                <span className="text-lg opacity-80">{link.icon}</span>
                <span className="font-medium">{t(link.key)}</span>
              </Link>
            ))}
          </nav>

          {/* Note: Lang/Theme settings are REMOVED from here as per request ("remove from there they will stay at header only") */}
        </div>
      </div>

    </header>
  );
};

export default Header;