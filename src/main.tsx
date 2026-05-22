import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Suppress TensorFlow Lite XNNPACK informational logs
const originalLog = console.log;
const originalInfo = console.info;
const originalWarn = console.warn;
const originalError = console.error;

const suppressXNNPACK = (originalMethod: any) => {
  return function (...args: any[]) {
    const msg = args[0];
    if (msg && typeof msg === 'string') {
      if (msg.includes('XNNPACK') || msg.includes('Failed to list models') || msg.includes('RpcError')) {
        return;
      }
    }
    originalMethod.apply(console, args);
  };
};

console.log = suppressXNNPACK(originalLog);
console.info = suppressXNNPACK(originalInfo);
console.warn = suppressXNNPACK(originalWarn);
console.error = suppressXNNPACK(originalError);

// Global error handlers to suppress specific internal errors from showing up in the UI/Console
window.addEventListener('error', (event) => {
  if (event.message && (event.message.includes('Failed to list models') || event.message.includes('RpcError'))) {
    event.preventDefault();
    event.stopPropagation();
  }
}, true);

window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason;
  const msg = reason?.message || (typeof reason === 'string' ? reason : '');
  if (msg && (msg.includes('Failed to list models') || msg.includes('RpcError'))) {
    event.preventDefault();
    event.stopPropagation();
  }
}, true);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
