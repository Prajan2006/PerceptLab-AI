import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

import App from './App';

import '@/assets/styles/tokens.css';
import '@/assets/styles/global.css';
import '@/assets/styles/layout.css';
import '@/assets/styles/components.css';
import '@/assets/styles/pages.css';
import '@/assets/styles/research.css';

// Apply the persisted (or OS-preferred) theme before first paint so the
// token system renders the correct palette immediately.
{
  const stored = localStorage.getItem('pl-theme');
  const theme =
    stored === 'dark' || stored === 'light'
      ? stored
      : window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
  document.documentElement.dataset.theme = theme;
}

const rootElement = document.getElementById('root');
if (rootElement === null) {
  throw new Error('Root element #root not found.');
}

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
