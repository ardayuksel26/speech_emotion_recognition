import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import "./tailwind.css";
import './i18n';
import { I18nextProvider } from 'react-i18next';
import i18n from './i18n';
import axios from 'axios';

axios.defaults.headers.common['ngrok-skip-browser-warning'] = 'true';


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <I18nextProvider i18n={i18n}>
      <App />
    </I18nextProvider>
  </StrictMode>,
)
